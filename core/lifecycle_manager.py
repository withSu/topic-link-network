"""
메모리 생명주기 관리 모듈
메모리의 승격, 강등, 망각 처리
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List

from models.enums import MemoryTier
from models.memory_entry import MemoryEntry
from core.memory_manager import MemoryManager


class LifecycleManager:
    """
    메모리 생명주기 관리자
    
    기능:
    - 메모리 계층 간 이동 관리
    - 망각 프로세스 실행
    - 시스템 자원 최적화
    """
    
    def __init__(self, memory_manager: MemoryManager, config: Dict[str, Any]):
        self.memory_manager = memory_manager
        self.config = config
        
        # 승격 규칙
        self.promotion_rules = {
            # 워킹 메모리 제거 (초단기 기억은 외부에서 처리)
            MemoryTier.SHORT_TERM: {
                'importance_threshold': 0.8,
                'access_threshold': 5,
                'age_threshold': timedelta(days=7)
            }
        }
        
        # 강등 규칙
        self.demotion_rules = {
            # 워킹 메모리 제거 (초단기 기억은 외부에서 처리)
            MemoryTier.SHORT_TERM: {
                'max_age': timedelta(days=30),
                'inactivity_threshold': timedelta(days=14)
            }
        }
        
        # 통계
        self.stats = {
            'promotion_count': 0,
            'demotion_count': 0,
            'forgotten_count': 0
        }
    
    async def run_lifecycle_cycle(self) -> None:
        """생명주기 사이클 실행"""
        # 비동기 작업들을 병렬로 실행
        tasks = [
            self._check_promotions(),
            self._check_demotions(),
            self._apply_decay()
        ]
        
        await asyncio.gather(*tasks)
    
    async def _check_promotions(self) -> None:
        """승격 검사"""
        # 단기 메모리 → 장기 메모리
        short_term_memories = await self.memory_manager.sqlite.find_by_tier(MemoryTier.SHORT_TERM)
        for memory in short_term_memories:
            if self._should_promote(memory, MemoryTier.SHORT_TERM):
                await self._promote_memory(memory, MemoryTier.LONG_TERM)
    
    def _should_promote(self, memory: MemoryEntry, from_tier: MemoryTier) -> bool:
        """승격 조건 확인"""
        rules = self.promotion_rules[from_tier]
        
        # 중요도 확인
        if memory.importance < rules['importance_threshold']:
            return False
        
        # 접근 횟수 확인
        if memory.access_count < rules['access_threshold']:
            return False
        
        # 생성 시간 확인
        if datetime.now() - memory.creation_time < rules['age_threshold']:
            return False
        
        return True
    
    async def _promote_memory(self, memory: MemoryEntry, to_tier: MemoryTier) -> None:
        """메모리 승격"""
        memory.tier = to_tier
        
        if to_tier == MemoryTier.LONG_TERM:
            await self.memory_manager.vector.save(memory)
            await self.memory_manager.sqlite.delete(memory.id)
        
        self.stats['promotion_count'] += 1
    
    async def _check_demotions(self) -> None:
        """강등 검사"""
        current_time = datetime.now()
        
        # 단기 메모리 검사
        short_term_memories = await self.memory_manager.sqlite.find_by_tier(MemoryTier.SHORT_TERM)
        for memory in short_term_memories:
            if self._should_demote(memory, current_time):
                await self._demote_memory(memory)
    
    def _should_demote(self, memory: MemoryEntry, current_time: datetime) -> bool:
        """강등 조건 확인"""
        rules = self.demotion_rules[memory.tier]
        
        # 최대 수명 확인
        if current_time - memory.creation_time > rules['max_age']:
            return True
        
        # 비활성화 시간 확인
        if memory.last_accessed and current_time - memory.last_accessed > rules['inactivity_threshold']:
            return True
        
        return False
    
    async def _demote_memory(self, memory: MemoryEntry) -> None:
        """메모리 강등 (삭제)"""
        await self.memory_manager.sqlite.delete(memory.id)
        self.stats['demotion_count'] += 1
        self.stats['forgotten_count'] += 1
    
    async def _apply_decay(self) -> None:
        """연관 강도 감소 적용"""
        # 연관 네트워크에도 감소 적용이 필요한 경우
        # association_network.apply_decay()를 호출
    """
    메모리 생명주기 관리 모듈 업데이트
    연결 강도 기반 메모리 승격 메커니즘 추가
    """

    async def _check_weak_connections(self) -> None:
        """
        약한 연결 강도를 가진 개념들을 장기 기억으로 승격
        """
        # 설정값
        connection_strength_threshold = 0.5  # 연결 강도 임계값
        
        # 약한 연결 강도 조회
        weak_connections = await self.memory_manager.sqlite.get_weak_connections(connection_strength_threshold)
        
        if weak_connections:
            print(f"{len(weak_connections)}개의 약한 연결을 발견했습니다. 장기 기억 승격 검토 중...")
            
            # 약한 연결의 개념들 수집
            weak_concepts = set()
            for conn in weak_connections:
                weak_concepts.add(conn['source'])
                weak_concepts.add(conn['target'])
            
            # 약한 개념들에 대한 메모리 승격
            promoted_count = 0
            for concept in weak_concepts:
                # 개념과 관련된 메모리 찾기
                memories = await self.memory_manager.sqlite.find_by_concepts([concept])
                
                for memory in memories:
                    # 중요도와 접근 횟수 기반 승격 결정
                    if memory.importance >= 0.7 or memory.access_count >= 3:
                        # 장기 기억으로 복사
                        memory.tier = MemoryTier.LONG_TERM
                        await self.memory_manager.vector.save(memory)
                        promoted_count += 1
            
            print(f"약한 연결 강도 기반 승격: {promoted_count}개 메모리가 장기 기억으로 승격되었습니다.")
            self.stats['promotion_count'] += promoted_count

    # lifecycle_manager.py의 run_lifecycle_cycle 메서드에 다음 코드 추가
    async def run_lifecycle_cycle(self) -> None:
        """생명주기 사이클 실행"""
        # 비동기 작업들을 병렬로 실행
        tasks = [
            self._check_promotions(),
            self._check_demotions(),
            self._apply_decay(),
            self._check_weak_connections()  # 약한 연결 확인 추가
        ]
        
        await asyncio.gather(*tasks)