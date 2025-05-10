"""
메모리 관리 통합 모듈
모든 메모리 계층을 통합하여 관리
"""
import asyncio
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime

from models.memory_entry import MemoryEntry
from models.enums import MemoryTier, ConnectionType
from storage.sqlite_storage import SQLiteStorage
from storage.vector_storage import VectorStorage
from utils.cache import LRUCache
from utils.bloom_filter import BloomFilter


class MemoryManager:
    """
    통합 메모리 관리자
    
    기능:
    - 계층화된 메모리 저장소 관리
    - 효율적인 검색 및 저장
    - 비동기 처리 통합
    """
    
    def __init__(self, config: Dict[str, Any]):
        # 저장소 초기화
        self.sqlite = SQLiteStorage(
            db_path=config['storage']['db_path']
        )
        
        self.vector = VectorStorage(
            db_path=config['storage']['chroma_path'],
            collection_name=config['storage']['collection_name'],
            embedding_model=config['embedding']['model']
        )
        
        # 캐시 및 인덱스
        self.search_cache = LRUCache(capacity=1000)
        self.bloom_filter = BloomFilter(capacity=100000, error_rate=0.001)
        
        # 설정값
        self.connection_strength_threshold = 0.5  # 연결 강도 임계값
        self.promotion_threshold = 0.8  # 장기 기억 승격 임계값
        
        # 통계
        self.stats = {
            'total_memories': 0,
            'search_operations': 0,
            'save_operations': 0,
            'tier_distribution': {tier: 0 for tier in MemoryTier}
        }
    
    async def save_memory(
        self,
        content: Dict[str, Any],
        concepts: List[str],
        importance: float = 0.5,
        emotional_weight: float = 0.0,
        tier: MemoryTier = MemoryTier.SHORT_TERM  # tier 매개변수 추가
    ) -> str:
        """메모리 저장"""
        memory_id = str(uuid.uuid4())
        
        # 메모리 엔트리 생성
        memory = MemoryEntry(
            id=memory_id,
            content=content,
            concepts=concepts,
            importance=importance,
            emotional_weight=emotional_weight,
            tier=tier  # 전달받은 tier 사용
        )
        
        # 티어에 따라 적절한 저장소에 저장
        if tier == MemoryTier.SHORT_TERM:
            await self.sqlite.save(memory)
        elif tier == MemoryTier.LONG_TERM:
            await self.vector.save(memory)
        
        # 블룸 필터 업데이트
        for concept in concepts:
            self.bloom_filter.add(concept)
        
        # 통계 업데이트
        self.stats['total_memories'] += 1
        self.stats['save_operations'] += 1
        self.stats['tier_distribution'][tier] += 1
        
        return memory_id
    
    async def process_external_keywords(self, keywords: List[str], content: Dict[str, Any], 
                                        importance: float = 0.5, emotional_weight: float = 0.0) -> str:
        """
        외부에서 받은 키워드를 처리하고 적절한 메모리에 저장
        
        Args:
            keywords: 외부에서 전달받은 키워드 목록
            content: 저장할 컨텐츠 데이터
            importance: 중요도 점수
            emotional_weight: 감정 가중치
            
        Returns:
            저장된 메모리 ID
        """
        # 키워드(컨셉트)들을 대상으로 메모리 저장
        return await self.save_memory(
            content=content,
            concepts=keywords,
            importance=importance,
            emotional_weight=emotional_weight,
            tier=MemoryTier.SHORT_TERM
        )
    
    async def _search_short_term(self, concepts: List[str]) -> List[MemoryEntry]:
        """단기 메모리 검색"""
        try:
            return await self.sqlite.find_by_concepts(concepts)
        except Exception as e:
            print(f"단기 메모리 검색 오류: {e}")
            return []

    async def _search_long_term(self, concepts: List[str], query_text: Optional[str] = None) -> List[Dict[str, Any]]:
        """장기 메모리 검색"""
        try:
            if query_text:
                return await self.vector.search_by_text(query_text, top_k=10)
            else:
                return await self.vector.search_by_concepts(concepts, top_k=10)
        except Exception as e:
            print(f"장기 메모리 검색 오류: {e}")
            return []

    async def search_memories(
        self,
        concepts: List[str],
        query_text: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """통합 메모리 검색"""
        # 캐시 확인
        cache_key = f"search:{':'.join(concepts)}"
        cached_result = self.search_cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # 병렬 검색 실행
        search_tasks = [
            self._search_short_term(concepts),
            self._search_long_term(concepts, query_text)
        ]
        
        results = await asyncio.gather(*search_tasks)
        
        # 연관 네트워크 검색
        network_results = {}
        if hasattr(self, 'network') and self.network:
            network_results = self.network.find_associations(concepts, depth=2)
        else:
            # 네트워크가 없는 경우 빈 딕셔너리 사용
            network_results = {}
        
        # 결과 병합 및 재순위
        merged_results = self._merge_search_results(results, network_results, concepts)
        
        # 캐시에 저장
        self.search_cache.put(cache_key, merged_results)
        
        # 통계 업데이트
        self.stats['search_operations'] += 1
        
        return merged_results
    
    def _merge_search_results(
        self,
        search_results: List[List],
        network_results: Dict,
        query_concepts: List[str]
    ) -> List[Dict]:
        # 로깅 추가
        print(f"Search results: {len(search_results)}")
        for i, result in enumerate(search_results):
            print(f"Result set {i}: {len(result)} items")
        print(f"Network results: {len(network_results)} items")
        all_results = []
        
        # SQLite 결과
        sqlite_results = search_results[0]
        for memory in sqlite_results:
            # 개념 매칭 점수 계산
            concept_match_score = 0
            if query_concepts:  # 빈 리스트 검사 추가
                concept_match_score = sum(1 for c in memory.concepts if c in query_concepts) / len(query_concepts)
            score = 0.8 * memory.importance + 0.2 * concept_match_score
            all_results.append({
                'memory': memory,
                'score': score,
                'source': 'short_term'
            })
        
        # Vector DB 결과
        vector_results = search_results[1]
        for result in vector_results:
            score = 0.6 * result['similarity']
            all_results.append({
                'memory': result,
                'score': score,
                'source': 'long_term'
            })
        
        # 점수 기반 정렬
        sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
        
        # 중복 제거
        unique_results = []
        seen_ids = set()
        
        for result in sorted_results:
            memory_id = result['memory'].id if hasattr(result['memory'], 'id') else result['memory']['id']
            if memory_id not in seen_ids:
                seen_ids.add(memory_id)
                unique_results.append(result)
        
        return unique_results[:10]  # 상위 10개만 반환
    
    async def save_topic_subtopic_relations(self, json_data: List[Dict[str, str]]) -> None:
        """
        JSON 데이터에서 추출한 토픽과 서브토픽 관계를 저장
        
        Args:
            json_data: 친구 시스템에서 제공하는 JSON 데이터
        """
        # 토픽과 서브토픽 관계 추출
        relations = {}
        for item in json_data:
            topic = item.get('topic')
            sub_topic = item.get('sub_topic')
            
            if topic and sub_topic:
                if topic not in relations:
                    relations[topic] = set()
                relations[topic].add(sub_topic)
        
        # 관계를 SQLite에 저장
        for topic, sub_topics in relations.items():
            for sub_topic in sub_topics:
                # 계층적 관계 저장 (토픽 -> 서브토픽)
                await self.sqlite.save_concept_connection(
                    source_concept=topic,
                    target_concept=sub_topic,
                    weight=0.9,  # 높은 가중치
                    connection_type='hierarchical'
                )
                
                # 서브토픽 간 관계 저장 (같은 토픽에 속한 서브토픽끼리)
                for other_sub in sub_topics:
                    if sub_topic != other_sub:
                        await self.sqlite.save_concept_connection(
                            source_concept=sub_topic,
                            target_concept=other_sub,
                            weight=0.7,  # 중간 가중치
                            connection_type='semantic'
                        )
        
        # 토픽 간 관계 저장 (추가됨)
        topics = list(relations.keys())
        for i, topic1 in enumerate(topics):
            for topic2 in topics[i+1:]:
                # 토픽 간의 연결 강도는 공유하는 서브토픽 수에 비례
                shared_subtopics = relations[topic1].intersection(relations[topic2])
                if shared_subtopics:
                    # 공유 서브토픽이 많을수록 강한 연결
                    connection_strength = min(0.5 + (len(shared_subtopics) * 0.1), 0.9)
                else:
                    # 공유 서브토픽이 없으면 약한 연결
                    connection_strength = 0.4
                
                await self.sqlite.save_concept_connection(
                    source_concept=topic1,
                    target_concept=topic2,
                    weight=connection_strength,
                    connection_type='semantic'
                )
                
                # 연결 강도가 임계값 이하인 경우 장기 기억으로 승격 검토
                if connection_strength <= self.connection_strength_threshold:
                    await self._check_for_promotion_to_long_term(topic1)
                    await self._check_for_promotion_to_long_term(topic2)
        
        # 메모에서 추출한 추가 키워드 관계 설정 (옵션)
        keywords_by_topic = {}
        for item in json_data:
            topic = item.get('topic')
            memo = item.get('memo', '')
            
            if topic and memo:
                # 간단한 키워드 추출 (실제로는 더 정교한 방법 사용 가능)
                keywords = [w for w in memo.split() if len(w) > 1]
                
                if topic not in keywords_by_topic:
                    keywords_by_topic[topic] = set()
                
                for keyword in keywords:
                    keywords_by_topic[topic].add(keyword)
        
        # 토픽과 메모 키워드 간 관계 저장
        for topic, keywords in keywords_by_topic.items():
            for keyword in keywords:
                await self.sqlite.save_concept_connection(
                    source_concept=topic,
                    target_concept=keyword,
                    weight=0.5,  # 낮은 가중치
                    connection_type='semantic'
                )
    
    async def _check_for_promotion_to_long_term(self, concept: str) -> None:
        """
        특정 개념을 장기 기억으로 승격할지 검토
        
        Args:
            concept: 검토할 개념
        """
        # 개념과 관련된 모든 메모리 찾기
        memories = await self.sqlite.find_by_concepts([concept])
        
        for memory in memories:
            # 중요도가 임계값 이상이면 승격
            if memory.importance >= self.promotion_threshold:
                # 장기 기억으로 복사
                memory.tier = MemoryTier.LONG_TERM
                await self.vector.save(memory)
                print(f"개념 '{concept}'와 관련된 메모리 '{memory.id}'가 장기 기억으로 승격되었습니다.")
    
    async def promote_weak_connections_to_long_term(self) -> None:
        """
        약한 연결 강도를 가진 개념들의 메모리를 장기 기억으로 승격
        """
        # 연결 강도가 임계값 이하인 모든 연결 가져오기
        weak_connections = await self.sqlite.get_weak_connections(self.connection_strength_threshold)
        
        # 약한 연결의 개념들 수집
        weak_concepts = set()
        for conn in weak_connections:
            weak_concepts.add(conn['source'])
            weak_concepts.add(conn['target'])
        
        # 각 개념의 메모리 검토 및 승격
        for concept in weak_concepts:
            await self._check_for_promotion_to_long_term(concept)
    
    async def get_stats(self) -> Dict[str, Any]:
        """통합 통계 반환"""
        # Redis 통계 제거
        sqlite_stats = self.sqlite.get_stats()
        vector_stats = self.vector.get_stats()
        
        return {
            'total_memories': self.stats['total_memories'],
            'tier_distribution': {
                'short_term': sqlite_stats['total_memories'],
                'long_term': vector_stats['total_memories']
            },
            'operations': {
                'search': self.stats['search_operations'],
                'save': self.stats['save_operations']
            },
            'resources': {
                'cache_size': len(self.search_cache.cache)
            }
        }