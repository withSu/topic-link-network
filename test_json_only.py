"""
JSON 데이터만으로 연관 네트워크 구축을 테스트하는 스크립트
"""
import asyncio
import json
import logging
from config.settings import SystemConfig
from chatbot.chatbot import RealtimeAssociativeChatbot
from models.enums import ConnectionType

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 테스트 데이터
TEST_JSON_DATA = [
  {
    "user_id": "test_user",
    "sub_topic": "name",
    "topic": "basic_info",
    "memo": "조현호"
  },
  {
    "topic": "basic_info",
    "user_id": "test_user",
    "sub_topic": "age",
    "memo": "29"
  },
  {
    "sub_topic": "foods",
    "topic": "interest",
    "user_id": "test_user",
    "memo": "사용자는 치킨과 소고기를 좋아함"
  },
  {
    "user_id": "test_user",
    "sub_topic": "previous_projects",
    "topic": "work",
    "memo": "어제(2025/05/03) 만든 프로그램 유지보수하느라 많은 시간 소모됨"
  },
  {
    "user_id": "test_user",
    "sub_topic": "psychological_state",
    "topic": "work",
    "memo": "프로그램 문제로 인해 스트레스 느낌, \"젠장\" 표현에서 불만과 피로감 추정"
  },
  {
    "sub_topic": "feelings",
    "topic": "psychological",
    "user_id": "test_user",
    "memo": "프로그램 문제가 생겨서 불만족감 표현 (\"젠장\")"
  },
  {
    "topic": "life_event",
    "user_id": "test_user",
    "sub_topic": "appointment",
    "memo": "사용자 약속이 2025/05/11에 취소된 것으로 추정됨 (내일)"
  },
  {
    "sub_topic": "social_activities",
    "user_id": "test_user",
    "topic": "psychological",
    "memo": "내일 예정된 약속이 취소된 것으로 보임"
  },
  {
    "sub_topic": "appointment_change",
    "topic": "life_event",
    "user_id": "test_user",
    "memo": "내일 약속 취소 가능성 있음 (2025/05/08)"
  }
]

async def test_json_processing():
    """JSON 데이터로 연관 강도 설정 테스트"""
    # 시스템 초기화
    config = SystemConfig()
    chatbot = RealtimeAssociativeChatbot(config)
    
    print("=== JSON 데이터만으로 연관 강도 설정 테스트 ===")
    
    # 1. 토픽과 서브토픽 추출
    user_id = "test_user"
    dummy_message = "테스트 메시지"  # 실제 메시지 내용은 무시됨
    
    # 2. 토픽과 서브토픽 관계 저장
    await chatbot.memory_manager.save_topic_subtopic_relations(TEST_JSON_DATA)
    
    # 3. 직접 연관 네트워크에 개념 추가 및 관계 설정
    topics = {}
    keywords = []
    
    # 토픽과 서브토픽 추출
    for item in TEST_JSON_DATA:
        if item.get('user_id') != user_id:
            continue
            
        topic = item.get('topic')
        sub_topic = item.get('sub_topic')
        
        if topic and topic not in keywords:
            keywords.append(topic)
            
        if sub_topic and sub_topic not in keywords:
            keywords.append(sub_topic)
            
        if topic and sub_topic:
            if topic not in topics:
                topics[topic] = []
            if sub_topic not in topics[topic]:
                topics[topic].append(sub_topic)
    
    # 개념 활성화
    for keyword in keywords:
        chatbot.association_network.activate_concept(keyword, keywords)
    
    # 연결 설정
    for topic, subtopics in topics.items():
        # 토픽과 서브토픽 간 연결
        for subtopic in subtopics:
            chatbot.association_network.connect_concepts(
                topic, 
                subtopic, 
                connection_type=ConnectionType.HIERARCHICAL,
                custom_strength=0.9
            )
            
            # 서브토픽 간 연결
            for other_subtopic in subtopics:
                if subtopic != other_subtopic:
                    chatbot.association_network.connect_concepts(
                        subtopic,
                        other_subtopic,
                        connection_type=ConnectionType.SEMANTIC,
                        custom_strength=0.7
                    )
    
    # 4. 결과 확인: 연관 네트워크 시각화
    print("\n결과 확인: 토픽 및 서브토픽")
    print(f"추출된 키워드: {keywords}")
    print(f"토픽 구조: {json.dumps(topics, indent=2, ensure_ascii=False)}")
    
    # 5. 연관 네트워크 시각화
    print("\n연관 네트워크 시각화 생성 중...")
    for topic in topics:
        viz_path = f"json_test_{topic}.png"
        result = await chatbot.visualize_associations(topic, viz_path)
        if result:
            print(f"'{topic}' 연관 네트워크 저장: {viz_path}")
    
    # 6. 연관 검색 테스트
    print("\n연관 검색 테스트:")
    for topic in topics:
        associations = chatbot.association_network.find_associations([topic])
        print(f"'{topic}'과 연관된 개념: {associations}")
    
    # 7. 종료
    await chatbot.shutdown()
    print("\n테스트 완료")

if __name__ == "__main__":
    asyncio.run(test_json_processing())