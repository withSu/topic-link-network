"""
토픽 간 및 토픽 내부 연결 강도 시각화 테스트 스크립트
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Set

from config.settings import SystemConfig
from chatbot.chatbot import RealtimeAssociativeChatbot
from models.enums import MemoryTier, ConnectionType

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 테스트 JSON 데이터
TEST_JSON_DATA = [
    {
        "user_id": "test_user",
        "sub_topic": "name",
        "topic": "basic_info",
        "memo": "홍길동"
    },
    {
        "user_id": "test_user",
        "sub_topic": "age",
        "topic": "basic_info",
        "memo": "25"
    },
    {
        "user_id": "test_user",
        "sub_topic": "foods",
        "topic": "interest",
        "memo": "사용자는 치킨과 소고기를 좋아함"
    },
    {
        "user_id": "test_user",
        "sub_topic": "previous_projects",
        "topic": "work",
        "memo": "프로그램 유지보수 작업"
    },
    {
        "user_id": "test_user",
        "sub_topic": "psychological_state",
        "topic": "work",
        "memo": "스트레스 받음"
    },
    {
        "user_id": "test_user",
        "sub_topic": "feelings",
        "topic": "psychological",
        "memo": "프로그램 문제로 불만족감"
    },
    {
        "user_id": "test_user",
        "sub_topic": "appointment",
        "topic": "life_event",
        "memo": "약속 취소됨"
    },
    {
        "user_id": "test_user",
        "sub_topic": "social_activities",
        "topic": "psychological",
        "memo": "약속 취소"
    },
    {
        "user_id": "test_user",
        "sub_topic": "appointment_change",
        "topic": "life_event",
        "memo": "약속 취소"
    }
]

# 테스트용 함수: 직접 그래프 구축
async def build_graph_directly(chatbot):
    """
    테스트를 위해 연관 네트워크 그래프를 직접 구축
    시각화 테스트를 위해 사용
    """
    # 토픽 목록 추출
    topics = set(item["topic"] for item in TEST_JSON_DATA)
    
    # 토픽별 서브토픽 매핑 수집
    topic_subtopics = {}
    for item in TEST_JSON_DATA:
        topic = item.get("topic")
        subtopic = item.get("sub_topic")
        if topic and subtopic:
            if topic not in topic_subtopics:
                topic_subtopics[topic] = []
            if subtopic not in topic_subtopics[topic]:
                topic_subtopics[topic].append(subtopic)
    
    # 1. 토픽을 노드로 추가
    for topic in topics:
        chatbot.association_network.add_concept(topic, {"type": "topic"})
    
    # 2. 서브토픽을 노드로 추가하고 토픽과 연결
    for topic, subtopics in topic_subtopics.items():
        for subtopic in subtopics:
            chatbot.association_network.add_concept(subtopic, {"type": "subtopic"})
            # 토픽 → 서브토픽 계층적 연결
            chatbot.association_network.connect_concepts(
                topic, 
                subtopic, 
                connection_type=ConnectionType.HIERARCHICAL,
                custom_strength=0.9
            )
    
    # 3. 토픽 간 연결 설정
    topic_list = list(topics)
    for i, topic1 in enumerate(topic_list):
        for topic2 in topic_list[i+1:]:
            # 공유 서브토픽 수에 기반한 연결 강도
            shared_subtopics = set(topic_subtopics.get(topic1, [])).intersection(
                set(topic_subtopics.get(topic2, []))
            )
            
            # 공유 서브토픽이 있으면 강한 연결, 없으면 약한 연결
            if shared_subtopics:
                strength = min(0.5 + len(shared_subtopics) * 0.1, 0.9)
            else:
                strength = 0.4
            
            # 양방향 연결 설정
            chatbot.association_network.connect_concepts(
                topic1, 
                topic2, 
                connection_type=ConnectionType.SEMANTIC,
                custom_strength=strength
            )
            chatbot.association_network.connect_concepts(
                topic2, 
                topic1, 
                connection_type=ConnectionType.SEMANTIC,
                custom_strength=strength
            )
    
    # 4. 같은 토픽의 서브토픽 간 연결
    for topic, subtopics in topic_subtopics.items():
        for i, subtopic1 in enumerate(subtopics):
            for subtopic2 in subtopics[i+1:]:
                # 같은 토픽 내 서브토픽 간의 의미적 연결
                chatbot.association_network.connect_concepts(
                    subtopic1,
                    subtopic2,
                    connection_type=ConnectionType.SEMANTIC,
                    custom_strength=0.7
                )
                chatbot.association_network.connect_concepts(
                    subtopic2,
                    subtopic1,
                    connection_type=ConnectionType.SEMANTIC,
                    custom_strength=0.7
                )
    
    # 5. 메모에서 키워드 추출 및 연결
    for item in TEST_JSON_DATA:
        topic = item.get("topic")
        subtopic = item.get("sub_topic")
        memo = item.get("memo", "")
        
        # 간단한 키워드 추출
        keywords = [word for word in memo.split() if len(word) > 1]
        
        # 키워드를 노드로 추가하고 토픽/서브토픽과 연결
        for keyword in keywords:
            chatbot.association_network.add_concept(keyword, {"type": "keyword"})
            
            # 토픽 → 키워드 연결
            chatbot.association_network.connect_concepts(
                topic,
                keyword,
                connection_type=ConnectionType.SEMANTIC,
                custom_strength=0.5
            )
            
            # 서브토픽 → 키워드 연결
            chatbot.association_network.connect_concepts(
                subtopic,
                keyword,
                connection_type=ConnectionType.SEMANTIC,
                custom_strength=0.6
            )

async def test_visualizations():
    """토픽 간 및 토픽 내부 연결 강도 시각화 테스트"""
    # 시스템 초기화
    config = SystemConfig()
    chatbot = RealtimeAssociativeChatbot(config)
    
    print("=== 토픽 간 및 토픽 내부 연결 강도 시각화 테스트 ===")
    
    # 1. 직접 그래프 구축 (시각화 테스트용)
    print("1. 테스트용 연관 네트워크 그래프 구축 중...")
    await build_graph_directly(chatbot)
    
    # 2. 토픽 간 연결 시각화
    print("\n2. 토픽 간 연결 시각화 생성 중...")
    # 토픽 목록 추출
    topics = set(item["topic"] for item in TEST_JSON_DATA)
    
    # 특수 시각화 - 모든 토픽 간 연결 보기
    all_topics_viz_path = "all_topics_network.png"
    result = await chatbot.visualize_associations("basic_info", all_topics_viz_path, 
                                               title_prefix="토픽 간 연결 네트워크")
    if result:
        print(f"모든 토픽 간 연결 네트워크 저장: {all_topics_viz_path}")
    
    # 3. 각 토픽별 내부 연결 시각화
    print("\n3. 각 토픽 내부 연결 시각화 생성 중...")
    for topic in topics:
        topic_viz_path = f"topic_internal_{topic}.png"
        result = await chatbot.visualize_associations(topic, topic_viz_path,
                                                  title_prefix=f"토픽 내부 연결 네트워크")
        if result:
            print(f"'{topic}' 내부 연결 네트워크 저장: {topic_viz_path}")
    
    # 4. 서브토픽 중심 시각화
    print("\n4. 서브토픽 중심 시각화 생성 중...")
    # 특정 서브토픽 선택
    selected_subtopics = ["foods", "feelings", "appointment", "previous_projects"]
    for subtopic in selected_subtopics:
        subtopic_viz_path = f"subtopic_{subtopic}.png"
        result = await chatbot.visualize_associations(subtopic, subtopic_viz_path,
                                                  title_prefix=f"서브토픽 중심 네트워크")
        if result:
            print(f"'{subtopic}' 중심 네트워크 저장: {subtopic_viz_path}")
    
    # 5. 전체 연관 네트워크 통계
    print("\n5. 연관 네트워크 통계:")
    network_stats = chatbot.association_network.get_network_stats()
    print(f"총 개념 수: {network_stats['total_concepts']}")
    print(f"총 연결 수: {network_stats['total_connections']}")
    print(f"평균 연결 수: {network_stats['average_connections']:.2f}")
    
    print("\n가장 많이 연결된 개념:")
    for concept, count in network_stats['most_connected']:
        print(f"  {concept}: {count}개 연결")
    
    print("\n가장 강한 연결:")
    for source, target, weight in network_stats['strongest_connections']:
        print(f"  {source} → {target}: 강도 {weight:.2f}")
    
    # 종료
    await chatbot.shutdown()
    print("\n테스트 완료")

if __name__ == "__main__":
    asyncio.run(test_visualizations())