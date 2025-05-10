"""
그래프 직접 구축 및 시각화 테스트
"""
import asyncio
import json
from datetime import datetime

from config.settings import SystemConfig
from chatbot.chatbot import RealtimeAssociativeChatbot
from models.enums import ConnectionType
from json_file_handler import load_json_topics_from_file, create_sample_json_file

async def build_and_visualize_graph():
    """
    연관 네트워크 그래프를 직접 구축하고 시각화
    """
    # 1. 시스템 초기화
    config = SystemConfig()
    chatbot = RealtimeAssociativeChatbot(config)
    
    print("=== 그래프 직접 구축 및 시각화 테스트 ===")
    
    # 2. 샘플 JSON 파일 생성 또는 로드
    json_file = create_sample_json_file("sample_for_graph.json")
    json_data = load_json_topics_from_file(json_file)
    
    print(f"JSON 파일 로드: {json_file}")
    print(f"데이터 항목 수: {len(json_data)}")
    
    # 3. 토픽 및 서브토픽 추출
    topics = {}
    for item in json_data:
        topic = item.get('topic')
        sub_topic = item.get('sub_topic')
        
        if topic and sub_topic:
            if topic not in topics:
                topics[topic] = []
            if sub_topic not in topics[topic]:
                topics[topic].append(sub_topic)
    
    print("\n토픽 및 서브토픽 구조:")
    for topic, subtopics in topics.items():
        print(f"- {topic}: {subtopics}")
    
    # 4. 그래프에 개념(노드) 추가
    print("\n그래프에 개념 추가 중...")
    
    # 4.1 토픽 추가
    for topic in topics:
        chatbot.association_network.add_concept(topic, {"type": "topic"})
    
    # 4.2 서브토픽 추가
    for topic, subtopics in topics.items():
        for subtopic in subtopics:
            chatbot.association_network.add_concept(subtopic, {"type": "subtopic"})
    
    # 5. 연결(엣지) 추가
    print("\n연결 추가 중...")
    
    # 5.1 토픽-서브토픽 계층적 연결
    for topic, subtopics in topics.items():
        for subtopic in subtopics:
            chatbot.association_network.connect_concepts(
                topic, 
                subtopic, 
                connection_type=ConnectionType.HIERARCHICAL,
                custom_strength=0.9
            )
    
    # 5.2 토픽 간 연결
    topic_list = list(topics.keys())
    for i, topic1 in enumerate(topic_list):
        for topic2 in topic_list[i+1:]:
            # 공유 서브토픽 수에 기반한 연결 강도
            shared_subtopics = set(topics[topic1]).intersection(set(topics[topic2]))
            
            if shared_subtopics:
                strength = min(0.5 + 0.1 * len(shared_subtopics), 0.9)
            else:
                strength = 0.4
            
            chatbot.association_network.connect_concepts(
                topic1, 
                topic2, 
                connection_type=ConnectionType.SEMANTIC,
                custom_strength=strength
            )
    
    # 5.3 같은 토픽의 서브토픽 간 연결
    for topic, subtopics in topics.items():
        for i, subtopic1 in enumerate(subtopics):
            for subtopic2 in subtopics[i+1:]:
                chatbot.association_network.connect_concepts(
                    subtopic1,
                    subtopic2,
                    connection_type=ConnectionType.SEMANTIC,
                    custom_strength=0.7
                )
    
    # 6. 그래프 통계 확인
    print("\n그래프 구축 완료!")
    print(f"총 개념(노드) 수: {chatbot.association_network.graph.number_of_nodes()}")
    print(f"총 연결(엣지) 수: {chatbot.association_network.graph.number_of_edges()}")
    
    # 모든 노드 출력
    print("\n그래프 내 모든 개념:")
    for node in chatbot.association_network.graph.nodes():
        print(f"- {node}")
    
    # 7. 시각화
    print("\n토픽 연결 시각화 중...")
    for topic in topics:
        viz_path = f"graph_topic_{topic}.png"
        result = await chatbot.visualize_associations(topic, viz_path)
        if result:
            print(f"- '{topic}' 시각화 저장: {viz_path}")
        else:
            print(f"- '{topic}' 시각화 실패!")
    
    # 8. 서브토픽 시각화
    print("\n서브토픽 연결 시각화 중...")
    for topic, subtopics in topics.items():
        for subtopic in subtopics[:1]:  # 각 토픽의 첫 번째 서브토픽만
            viz_path = f"graph_subtopic_{subtopic}.png"
            result = await chatbot.visualize_associations(subtopic, viz_path)
            if result:
                print(f"- '{subtopic}' 시각화 저장: {viz_path}")
            else:
                print(f"- '{subtopic}' 시각화 실패!")
    
    # 9. 종료
    await chatbot.shutdown()
    print("\n테스트 완료!")

if __name__ == "__main__":
    asyncio.run(build_and_visualize_graph())