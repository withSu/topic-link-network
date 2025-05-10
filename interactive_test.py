"""
개선된 대화형 JSON 처리 및 시각화 테스트 스크립트
외부 JSON 파일을 로드하여 토픽/서브토픽 관계 설정 및 시각화
그래프 직접 구축 기능 추가
"""
import asyncio
import json
from datetime import datetime
import os
import argparse

from config.settings import SystemConfig
from chatbot.chatbot import RealtimeAssociativeChatbot
from models.enums import ConnectionType
from json_file_handler import load_json_topics_from_file, create_sample_json_file

# 그래프 직접 구축
async def build_graph_directly(chatbot, json_data):
    """
    연관 네트워크 그래프를 직접 구축
    """
    # 1. 토픽 및 서브토픽 추출
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
    
    # 2. 그래프에 개념(노드) 추가
    print("\n그래프에 개념 추가 중...")
    
    # 2.1 토픽 추가
    for topic in topics:
        chatbot.association_network.add_concept(topic, {"type": "topic"})
    
    # 2.2 서브토픽 추가
    for topic, subtopics in topics.items():
        for subtopic in subtopics:
            chatbot.association_network.add_concept(subtopic, {"type": "subtopic"})
    
    # 3. 연결(엣지) 추가
    print("\n연결 추가 중...")
    
    # 3.1 토픽-서브토픽 계층적 연결
    for topic, subtopics in topics.items():
        for subtopic in subtopics:
            chatbot.association_network.connect_concepts(
                topic, 
                subtopic, 
                connection_type=ConnectionType.HIERARCHICAL,
                custom_strength=0.9
            )
    
    # 3.2 토픽 간 연결
    topic_list = list(topics.keys())
    for i, topic1 in enumerate(topic_list):
        for topic2 in topic_list[i+1:]:
            # 공유 서브토픽 수에 기반한 연결 강도
            shared_subtopics = set(topics[topic1]).intersection(set(topics[topic2]))
            
            if shared_subtopics:
                strength = min(0.5 + 0.1 * len(shared_subtopics), 0.9)
            else:
                strength = 0.4
            
            # 양방향 연결
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
    
    # 3.3 같은 토픽의 서브토픽 간 연결
    for topic, subtopics in topics.items():
        for i, subtopic1 in enumerate(subtopics):
            for subtopic2 in subtopics[i+1:]:
                # 양방향 연결
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
    
    # 4. 메모 키워드 추출 및 연결
    for item in json_data:
        topic = item.get('topic')
        sub_topic = item.get('sub_topic')
        memo = item.get('memo', '')
        
        # 간단한 키워드 추출
        keywords = [word for word in memo.split() if len(word) > 1]
        
        # 최대 3개의 중요 키워드만 사용
        for keyword in keywords[:3]:
            # 키워드를 개념으로 추가
            chatbot.association_network.add_concept(keyword, {"type": "keyword"})
            
            # 토픽-키워드 연결
            if topic:
                chatbot.association_network.connect_concepts(
                    topic, 
                    keyword,
                    connection_type=ConnectionType.SEMANTIC,
                    custom_strength=0.5
                )
            
            # 서브토픽-키워드 연결
            if sub_topic:
                chatbot.association_network.connect_concepts(
                    sub_topic,
                    keyword,
                    connection_type=ConnectionType.SEMANTIC,
                    custom_strength=0.6
                )
    
    # 5. 그래프 통계 확인
    print("\n그래프 구축 완료!")
    print(f"총 개념(노드) 수: {chatbot.association_network.graph.number_of_nodes()}")
    print(f"총 연결(엣지) 수: {chatbot.association_network.graph.number_of_edges()}")
    
    return topics

async def run_enhanced_test(json_file_path: str = None):
    """
    대화형 JSON 처리 및 시각화 테스트 (개선됨)
    
    Args:
        json_file_path: JSON 파일 경로 (없으면 샘플 생성)
    """
    # 1. JSON 파일 확인 및 로드
    if not json_file_path or not os.path.exists(json_file_path):
        print("JSON 파일이 지정되지 않았거나 찾을 수 없습니다. 샘플 파일을 생성합니다.")
        json_file_path = create_sample_json_file()
    
    json_data = load_json_topics_from_file(json_file_path)
    if not json_data:
        print("JSON 데이터를 로드할 수 없습니다. 프로그램을 종료합니다.")
        return
    
    # 2. 시스템 초기화
    config = SystemConfig()
    chatbot = RealtimeAssociativeChatbot(config)
    
    print(f"=== 대화형 JSON 처리 테스트 (개선됨) ===")
    print(f"JSON 파일: {json_file_path}")
    print(f"로드된 데이터: {len(json_data)}개 항목")
    
    # 토픽 목록 추출
    topics = set(item["topic"] for item in json_data)
    print(f"토픽 목록: {topics}")
    
    # 그래프 구축 완료 여부
    graph_built = False
    
    # 3. 대화형 메뉴
    while True:
        print("\n==== 메뉴 ====")
        print("1. 네트워크 그래프 직접 구축 (개선된 방식)")
        print("2. 토픽/서브토픽 관계 설정 (기존 방식)")
        print("3. 토픽 간 연결 시각화")
        print("4. 특정 토픽 내부 연결 시각화")
        print("5. 약한 연결 찾기 및 메모리 승격")
        print("6. 모든 토픽 연결 통계 보기")
        print("7. 그래프 내 모든 개념 보기")
        print("8. JSON 데이터 보기")
        print("0. 종료")
        
        choice = input("\n선택: ").strip()
        
        if choice == "0":
            break
            
        elif choice == "1":
            # 그래프 직접 구축
            print("\n네트워크 그래프 직접 구축 중...")
            topic_dict = await build_graph_directly(chatbot, json_data)
            graph_built = True
            print("그래프 구축 완료!")
            
        elif choice == "2":
            # 토픽/서브토픽 관계 설정 (기존 방식)
            print("\n토픽/서브토픽 관계 설정 중...")
            await chatbot.memory_manager.save_topic_subtopic_relations(json_data)
            graph_built = True
            print("관계 설정 완료!")
            
        elif choice == "3":
            # 토픽 간 연결 시각화
            if not graph_built:
                print("\n먼저 그래프를 구축해야 합니다. 메뉴 1 또는 2를 선택하세요.")
                continue
            
            print("\n토픽 간 연결 시각화...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 시작 토픽 선택
            print(f"토픽 목록: {topics}")
            start_topic = input("시작 토픽을 선택하세요 (기본: basic_info): ").strip().lower()
            if not start_topic:
                start_topic = "basic_info"
                
            if start_topic not in [t.lower() for t in topics]:
                print(f"경고: '{start_topic}'는 유효한 토픽이 아닙니다. 기본값을 사용합니다.")
                start_topic = next(iter(topics))
            else:
                # 대소문자 일치 찾기
                for t in topics:
                    if t.lower() == start_topic:
                        start_topic = t
                        break
            
            viz_path = f"topics_connections_{start_topic}_{timestamp}.png"
            result = await chatbot.visualize_associations(start_topic, viz_path)
            if result:
                print(f"토픽 간 연결 네트워크가 저장되었습니다: {viz_path}")
            else:
                print("시각화 실패! 그래프에 해당 개념이 없거나 문제가 있습니다.")
            
        elif choice == "4":
            # 특정 토픽 내부 연결 시각화
            if not graph_built:
                print("\n먼저 그래프를 구축해야 합니다. 메뉴 1 또는 2를 선택하세요.")
                continue
                
            print("\n특정 토픽 내부 연결 시각화...")
            print(f"토픽 목록: {topics}")
            topic = input("토픽을 선택하세요: ").strip().lower()
            
            if not topic:
                print("유효한 토픽을 선택해주세요.")
                continue
                
            # 대소문자 일치 찾기
            found = False
            for t in topics:
                if t.lower() == topic:
                    topic = t  # 실제 대소문자 사용
                    found = True
                    break
                    
            if not found:
                print("유효한 토픽을 선택해주세요.")
                continue
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_path = f"topic_internal_{topic}_{timestamp}.png"
            
            result = await chatbot.visualize_associations(topic, viz_path)
            if result:
                print(f"토픽 내부 연결 네트워크가 저장되었습니다: {viz_path}")
            else:
                print("시각화 실패! 그래프에 해당 개념이 없거나 문제가 있습니다.")
            
        elif choice == "5":
            # 약한 연결 찾기 및 메모리 승격
            threshold = input("연결 강도 임계값을 입력하세요 (0.0~1.0, 기본: 0.5): ").strip()
            try:
                threshold = float(threshold) if threshold else 0.5
                if threshold < 0.0 or threshold > 1.0:
                    raise ValueError("임계값은 0.0에서 1.0 사이여야 합니다.")
            except ValueError as e:
                print(f"오류: {e}")
                threshold = 0.5
            
            print(f"\n연결 강도 {threshold} 이하의 약한 연결 찾는 중...")
            weak_connections = await chatbot.memory_manager.sqlite.get_weak_connections(threshold)
            
            print(f"약한 연결: {len(weak_connections)}개 발견")
            if weak_connections:
                for i, conn in enumerate(weak_connections[:10], 1):  # 처음 10개만 출력
                    print(f"{i}. {conn['source']} → {conn['target']}: 강도 {conn['weight']:.2f}")
                
                if len(weak_connections) > 10:
                    print(f"... 외 {len(weak_connections) - 10}개 더 있음")
                
                # 승격 여부 확인
                promote = input("\n이 약한 연결의 메모리를 장기 기억으로 승격하시겠습니까? (y/n): ").strip().lower()
                if promote == 'y':
                    print("\n약한 연결 기반 메모리 승격 중...")
                    await chatbot.memory_manager.promote_weak_connections_to_long_term()
                    print("승격 완료!")
            
        elif choice == "6":
            # 모든 토픽 연결 통계 보기
            if not graph_built:
                print("\n먼저 그래프를 구축해야 합니다. 메뉴 1 또는 2를 선택하세요.")
                continue
                
            print("\n토픽 연결 통계:")
            
            for topic in topics:
                connections = await chatbot.memory_manager.sqlite.get_concept_connections(topic)
                print(f"\n토픽 '{topic}'의 연결 ({len(connections)}개):")
                
                # 연결 강도별 분류
                strong = [c for c in connections if c['weight'] > 0.7]
                medium = [c for c in connections if 0.4 <= c['weight'] <= 0.7]
                weak = [c for c in connections if c['weight'] < 0.4]
                
                print(f"  - 강한 연결 ({len(strong)}개)")
                for conn in strong[:3]:  # 최대 3개만 출력
                    print(f"    → {conn['target']}: 강도 {conn['weight']:.2f} ({conn['type']})")
                
                print(f"  - 중간 연결 ({len(medium)}개)")
                for conn in medium[:3]:  # 최대 3개만 출력
                    print(f"    → {conn['target']}: 강도 {conn['weight']:.2f} ({conn['type']})")
                
                print(f"  - 약한 연결 ({len(weak)}개)")
                for conn in weak[:3]:  # 최대 3개만 출력
                    print(f"    → {conn['target']}: 강도 {conn['weight']:.2f} ({conn['type']})")
                    
        elif choice == "7":
            # 그래프 내 모든 개념 보기
            if not graph_built:
                print("\n먼저 그래프를 구축해야 합니다. 메뉴 1 또는 2를 선택하세요.")
                continue
                
            print("\n그래프 내 모든 개념:")
            nodes = list(chatbot.association_network.graph.nodes())
            node_count = len(nodes)
            
            if node_count == 0:
                print("그래프에 개념이 없습니다!")
            else:
                for i, node in enumerate(nodes, 1):
                    print(f"{i}. {node}")
                    
                print(f"\n총 {node_count}개 개념")
        
        elif choice == "8":
            # JSON 데이터 보기
            print("\nJSON 데이터:")
            for i, item in enumerate(json_data[:5], 1):  # 처음 5개만 출력
                print(f"{i}. {json.dumps(item, ensure_ascii=False)}")
            
            if len(json_data) > 5:
                print(f"... 외 {len(json_data) - 5}개 더 있음")
            
        else:
            print("잘못된 선택입니다. 다시 시도해주세요.")
    
    # 4. 종료
    await chatbot.shutdown()
    print("\n프로그램 종료")

if __name__ == "__main__":
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="개선된 대화형 JSON 처리 및 시각화 테스트")
    parser.add_argument('-f', '--file', help="JSON 파일 경로")
    args = parser.parse_args()
    
    # 테스트 실행
    asyncio.run(run_enhanced_test(args.file))