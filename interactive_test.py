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
        print("3. 모든 연결 시각화")  # 변경됨 - 모든 토픽, 서브토픽, 연결 시각화 (전체 그래프)
        print("4. 토픽 간 연결 시각화")    # 변경됨 - 토픽 간 연결만 시각화 (토픽 노드만)
        print("5. 특정 토픽 내부 연결 시각화")  # 변경됨 - 특정 토픽과 그 서브토픽만 시각화
        print("6. 약한 연결 찾기 및 메모리 승격")  
        print("7. 모든 토픽 연결 통계 보기")      
        print("8. 그래프 내 모든 개념 보기")      
        print("9. JSON 데이터 보기")             
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
        # 3번 메뉴 수정
        elif choice == "3":
            # 모든 토픽, 서브토픽 및 연결 시각화 (전체 그래프)
            if not graph_built:
                print("\n먼저 그래프를 구축해야 합니다. 메뉴 1 또는 2를 선택하세요.")
                continue
            
            print("\n모든 연결 시각화...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_path = f"all_connections_{timestamp}.png"
            
            # 전체 그래프 시각화 (show=False로 설정하여 두 번 표시되는 것 방지)
            result = await chatbot.visualize_all_connections(viz_path)
            if result:
                print(f"모든 연결 네트워크가 저장되었습니다: {viz_path}")
                # 이미지 파일 열기
                try:
                    import platform
                    if platform.system() == 'Windows':
                        os.startfile(viz_path)
                    elif platform.system() == 'Darwin':  # macOS
                        import subprocess
                        subprocess.run(['open', viz_path])
                    else:  # Linux
                        import subprocess
                        subprocess.run(['xdg-open', viz_path])
                    print("이미지 파일을 열었습니다.")
                except Exception as e:
                    print(f"이미지 파일을 자동으로 열 수 없습니다: {e}")
            else:
                print("시각화 실패! 그래프에 노드가 없거나 문제가 있습니다.")

        # 4번 메뉴 수정
        elif choice == "4":
            # 토픽 간 연결 시각화 (토픽 노드만)
            if not graph_built:
                print("\n먼저 그래프를 구축해야 합니다. 메뉴 1 또는 2를 선택하세요.")
                continue
            
            print("\n토픽 간 연결 시각화...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_path = f"topics_network_{timestamp}.png"
            
            # 토픽 간 연결 시각화 함수 호출 (show=False로 설정)
            result = await chatbot.visualize_topics_network(viz_path)
            if result:
                print(f"토픽 간 연결 네트워크가 저장되었습니다: {viz_path}")
                # 이미지 파일 열기
                try:
                    import platform
                    if platform.system() == 'Windows':
                        os.startfile(viz_path)
                    elif platform.system() == 'Darwin':  # macOS
                        import subprocess
                        subprocess.run(['open', viz_path])
                    else:  # Linux
                        import subprocess
                        subprocess.run(['xdg-open', viz_path])
                    print("이미지 파일을 열었습니다.")
                except Exception as e:
                    print(f"이미지 파일을 자동으로 열 수 없습니다: {e}")
            else:
                print("시각화 실패! 그래프에 토픽이 없거나 문제가 있습니다.")

        # 5번 메뉴 수정
        elif choice == "5":
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
            
            # 특정 토픽과 그 서브토픽 시각화 (show=False로 설정)
            result = await chatbot.visualize_topic_internal(topic, viz_path)
            if result:
                print(f"토픽 내부 연결 네트워크가 저장되었습니다: {viz_path}")
                # 이미지 파일 열기
                try:
                    import platform
                    if platform.system() == 'Windows':
                        os.startfile(viz_path)
                    elif platform.system() == 'Darwin':  # macOS
                        import subprocess
                        subprocess.run(['open', viz_path])
                    else:  # Linux
                        import subprocess
                        subprocess.run(['xdg-open', viz_path])
                    print("이미지 파일을 열었습니다.")
                except Exception as e:
                    print(f"이미지 파일을 자동으로 열 수 없습니다: {e}")
            else:
                print("시각화 실패! 그래프에 해당 토픽이 없거나 문제가 있습니다.")
        elif choice == "6":
            # 약한 연결 찾기 및 메모리 승격
            if not graph_built:
                print("\n먼저 그래프를 구축해야 합니다. 메뉴 1 또는 2를 선택하세요.")
                continue
                
            print("\n약한 연결 찾기 및 메모리 승격...")
            threshold = float(input("연결 강도 임계값을 입력하세요 (0.1-0.9, 기본: 0.5): ") or "0.5")
            
            # 약한 연결 조회
            weak_connections = await chatbot.memory_manager.sqlite.get_weak_connections(threshold)
            
            if not weak_connections:
                print(f"연결 강도 {threshold} 이하의 약한 연결이 없습니다.")
                continue
                
            print(f"\n연결 강도 {threshold} 이하의 약한 연결들:")
            for i, conn in enumerate(weak_connections[:10], 1):  # 최대 10개까지 출력
                print(f"{i}. {conn['source']} → {conn['target']} (강도: {conn['weight']:.2f}, 유형: {conn['type']})")
            
            if len(weak_connections) > 10:
                print(f"... 외 {len(weak_connections) - 10}개")
                
            promote = input("\n약한 연결 관련 메모리를 장기 기억으로 승격하시겠습니까? (y/n): ").lower() == 'y'
            
            if promote:
                # 메모리 승격
                result = await chatbot.memory_manager.promote_weak_connections_to_long_term(threshold)
                if result and result > 0:
                    print(f"{result}개의 메모리를 장기 기억으로 승격했습니다.")
                else:
                    print("승격할 메모리가 없습니다.")
        elif choice == "7":
            # 모든 토픽 연결 통계 보기
            if not graph_built:
                print("\n먼저 그래프를 구축해야 합니다. 메뉴 1 또는 2를 선택하세요.")
                continue
                
            print("\n모든 토픽 연결 통계:")
            
            # 모든 토픽 가져오기
            all_topics = []
            for node, data in chatbot.association_network.graph.nodes(data=True):
                metadata = data.get('metadata', {})
                if isinstance(metadata, dict) and metadata.get('type') == 'topic':
                    all_topics.append(node)
            
            # 각 토픽에 대한 통계 계산
            for topic in all_topics:
                # 연결 가져오기
                connections = list(chatbot.association_network.graph.edges(topic, data=True))
                
                if not connections:
                    print(f"- {topic}: 연결 없음")
                    continue
                    
                # 연결 강도별 분류
                strong_conn = [c for _, _, c in connections if c.get('weight', 0) >= 0.7]
                medium_conn = [c for _, _, c in connections if 0.4 <= c.get('weight', 0) < 0.7]
                weak_conn = [c for _, _, c in connections if c.get('weight', 0) < 0.4]
                
                print(f"- {topic}: 총 {len(connections)}개 연결")
                print(f"  * 강한 연결 (≥0.7): {len(strong_conn)}개")
                print(f"  * 중간 연결 (0.4~0.7): {len(medium_conn)}개")
                print(f"  * 약한 연결 (<0.4): {len(weak_conn)}개")
                
                # 예시 연결 표시
                if strong_conn:
                    sample = strong_conn[0]
                    print(f"    - 강한 연결 예시: → ... (강도: {sample.get('weight', 0):.2f})")
                if medium_conn:
                    sample = medium_conn[0]
                    print(f"    - 중간 연결 예시: → ... (강도: {sample.get('weight', 0):.2f})")
                if weak_conn:
                    sample = weak_conn[0]
                    print(f"    - 약한 연결 예시: → ... (강도: {sample.get('weight', 0):.2f})")
                
                print("")  # 줄바꿈
        elif choice == "8":
            # 그래프 내 모든 개념 보기
            if not graph_built:
                print("\n먼저 그래프를 구축해야 합니다. 메뉴 1 또는 2를 선택하세요.")
                continue
                
            print("\n그래프 내 모든 개념 (노드):")
            
            all_nodes = list(chatbot.association_network.graph.nodes(data=True))
            for i, (node, data) in enumerate(all_nodes, 1):
                node_type = data.get('metadata', {}).get('type', '일반')
                print(f"{i}. {node} ({node_type})")
                
            print(f"\n총 {len(all_nodes)}개의 개념이 있습니다.")
        elif choice == "9":
            # JSON 데이터 보기
            print("\nJSON 데이터:")
            
            # 처음 5개 항목만 출력
            for i, item in enumerate(json_data[:5], 1):
                print(f"{i}. {json.dumps(item, ensure_ascii=False)}")
                
            if len(json_data) > 5:
                print(f"\n... 외 {len(json_data) - 5}개 항목")
            
            print(f"\n총 {len(json_data)}개 항목이 있습니다.")
        else:
            print("잘못된 선택입니다. 다시 시도하세요.")
    
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