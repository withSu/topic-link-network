"""
실시간 연관 기억 챗봇 실행 예제
친구 시스템과의 통합 기능 포함
"""
import asyncio
import logging
import json
from datetime import datetime
from typing import List, Dict, Any

from config.settings import SystemConfig
from chatbot.chatbot import RealtimeAssociativeChatbot
from models.enums import MemoryTier, ConnectionType

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# 친구 시스템 통합 클래스
class FriendSystemAPI:
    """친구 시스템과의 통합을 위한 API 래퍼"""
    
    def __init__(self, api_url: str = None, api_key: str = None):
        self.api_url = api_url
        self.api_key = api_key
        # 샘플 데이터 - 실제로는 API 응답에서 받아옴
        self.sample_data = [
            {
                "user_id": "user1",
                "sub_topic": "name",
                "topic": "basic_info",
                "memo": "조현호"
            },
            {
                "topic": "basic_info",
                "user_id": "user1",
                "sub_topic": "age",
                "memo": "29"
            },
            {
                "sub_topic": "foods",
                "topic": "interest",
                "user_id": "user1",
                "memo": "사용자는 치킨과 소고기를 좋아함"
            },
            # ... 나머지 데이터
        ]
    
    async def analyze_message(self, user_id: str, message: str) -> List[Dict[str, Any]]:
        """
        메시지 분석 API 호출 (실제로는 외부 API 호출)
        
        Args:
            user_id: 사용자 ID
            message: 사용자 메시지
            
        Returns:
            토픽/서브토픽 JSON 데이터
        """
        # 실제로는 외부 API 호출하여 데이터 받아옴
        # import aiohttp
        # async with aiohttp.ClientSession() as session:
        #     async with session.post(
        #         f"{self.api_url}/analyze",
        #         json={"user_id": user_id, "message": message},
        #         headers={"Authorization": f"Bearer {self.api_key}"}
        #     ) as response:
        #         if response.status == 200:
        #             return await response.json()
        
        # 테스트를 위해 샘플 데이터 반환
        # 실제 상황에 맞게 간단한 필터링 수행
        filtered_data = []
        
        # 메시지 내용에 따라 적절한 토픽 선택
        if "음식" in message or "치킨" in message or "소고기" in message:
            # 음식 관련 토픽 필터링
            filtered_data.extend([item for item in self.sample_data if item.get("topic") == "interest" and item.get("sub_topic") == "foods"])
        
        if "스트레스" in message or "피곤" in message or "힘들" in message:
            # 심리 상태 관련 토픽 필터링
            filtered_data.extend([item for item in self.sample_data if item.get("topic") == "psychological"])
        
        if "약속" in message or "취소" in message or "일정" in message:
            # 일정 관련 토픽 필터링
            filtered_data.extend([item for item in self.sample_data if item.get("topic") == "life_event"])
        
        # 기본 정보는 항상 포함
        filtered_data.extend([item for item in self.sample_data if item.get("topic") == "basic_info"])
        
        # 결과가 없으면 기본 항목 추가
        if not filtered_data:
            filtered_data.append({
                "user_id": user_id,
                "topic": "conversation",
                "sub_topic": "general",
                "memo": f"일반 대화: '{message}'"
            })
        
        return filtered_data


# 테스트용 JSON 데이터 (실제로는 친구 시스템에서 제공)
SAMPLE_JSON_DATA = [
  {
    "user_id": "user1",
    "sub_topic": "name",
    "topic": "basic_info",
    "memo": "조현호"
  },
  {
    "topic": "basic_info",
    "user_id": "user1",
    "sub_topic": "age",
    "memo": "29"
  },
  {
    "sub_topic": "foods",
    "topic": "interest",
    "user_id": "user1",
    "memo": "사용자는 치킨과 소고기를 좋아함"
  },
  {
    "user_id": "user1",
    "sub_topic": "previous_projects",
    "topic": "work",
    "memo": "어제(2025/05/03) 만든 프로그램 유지보수하느라 많은 시간 소모됨"
  },
  {
    "user_id": "user1",
    "sub_topic": "psychological_state",
    "topic": "work",
    "memo": "프로그램 문제로 인해 스트레스 느낌, \"젠장\" 표현에서 불만과 피로감 추정"
  },
  {
    "sub_topic": "feelings",
    "topic": "psychological",
    "user_id": "user1",
    "memo": "프로그램 문제가 생겨서 불만족감 표현 (\"젠장\")"
  },
  {
    "topic": "life_event",
    "user_id": "user1",
    "sub_topic": "appointment",
    "memo": "사용자 약속이 2025/05/11에 취소된 것으로 추정됨 (내일)"
  },
  {
    "sub_topic": "social_activities",
    "user_id": "user1",
    "topic": "psychological",
    "memo": "내일 예정된 약속이 취소된 것으로 보임"
  },
  {
    "sub_topic": "appointment_change",
    "topic": "life_event",
    "user_id": "user1",
    "memo": "내일 약속 취소 가능성 있음 (2025/05/08)"
  }
]


async def run_demo():
    """데모 실행"""
    # 설정 로드
    config = SystemConfig()
    
    # 챗봇 초기화
    chatbot = RealtimeAssociativeChatbot(config)
    
    # 친구 시스템 API 초기화
    friend_api = FriendSystemAPI(api_url="https://friend-system-api.example.com", api_key="dummy_key")
    
    print("=== 실시간 연관 기억 챗봇 데모 (친구 시스템 통합) ===")
    print("마지막 문장에 '종료'를 입력하시면 시스템이 종료됩니다.")
    print("='시각화 <개념>'을 입력하시면 해당 개념의 연관 네트워크를 시각화합니다.")
    print("-" * 50)
    
    # 테스트 사용자 ID
    user_id = "demo_user"
    
    # 대화 루프
    try:
        while True:
            # 사용자 입력 받기
            user_input = input("\n사용자: ")
            
            # 종료 명령 처리
            if user_input.lower() == '종료':
                print("\n채팅을 종료합니다.")
                break
            
            # 시각화 명령 처리
            if user_input.startswith('시각화 '):
                concept = user_input.replace('시각화 ', '').strip()
                timestamp = int(datetime.now().timestamp())
                save_path = f"visualization_{concept}_{timestamp}.png"
                
                result = await chatbot.visualize_associations(concept, save_path)
                if result:
                    print(f"챗봇: 연관 네트워크를 {save_path}에 저장했습니다.")
                else:
                    print(f"챗봇: '{concept}'에 대한 연관 정보를 찾을 수 없습니다.")
                continue
            
            try:
                # 친구 시스템에서 JSON 데이터 받기
                # 실제로는 API 호출, 테스트를 위해 고정 샘플 사용
                # json_data = await friend_api.analyze_message(user_id, user_input)
                json_data = SAMPLE_JSON_DATA  # 테스트를 위한 샘플 데이터
                
                # 디버깅용 출력
                print(f"[DEBUG] 친구 시스템에서 받은 JSON 데이터:")
                print(json.dumps(json_data[:3], indent=2, ensure_ascii=False))  # 처음 3개만 출력
                print(f"... 총 {len(json_data)}개 항목")
                
                # JSON 데이터 처리
                response = await chatbot.process_json_keywords(user_id, user_input, json_data)
                print(f"챗봇: {response}")
                
            except Exception as e:
                print(f"[ERROR] 처리 중 오류 발생: {e}")
                # 백업으로 기존 메서드 사용
                keywords = user_input.split()
                response = await chatbot.chat(user_id, user_input, external_keywords=keywords)
                print(f"챗봇 (백업): {response}")
            
            # 주기적 통계 출력 (10번째 상호작용마다)
            stats = await chatbot.get_system_stats()
            if stats['chatbot']['total_interactions'] % 10 == 0:
                print("\n[시스템 통계]")
                print(f"- 총 상호작용: {stats['chatbot']['total_interactions']}회")
                print(f"- 평균 응답 시간: {stats['chatbot']['average_response_time']:.3f}초")
                print(f"- 활성 세션: {stats['active_sessions']}개")
                print("-" * 30)
    
    except KeyboardInterrupt:
        print("\n\n시스템을 종료합니다...")
    
    finally:
        # 시스템 종료
        await chatbot.shutdown()
        
        # 최종 통계 출력
        final_stats = await chatbot.get_system_stats()
        print("\n=== 최종 시스템 통계 ===")
        print(json.dumps(final_stats, indent=2, ensure_ascii=False))


async def run_json_test():
    """JSON 통합 테스트"""
    # 설정 로드
    config = SystemConfig()
    
    # 챗봇 초기화
    chatbot = RealtimeAssociativeChatbot(config)
    
    print("=== JSON 통합 테스트 ===")
    user_id = "test_user"
    
    # 테스트 시나리오: 작업 스트레스와 약속 취소
    user_input = "프로그램 유지보수하느라 스트레스 받았어. 내일 약속도 취소됐고..."
    
    print(f"\n사용자: {user_input}")
    
    # 실제 JSON 데이터 사용
    json_data = SAMPLE_JSON_DATA
    
    print(f"[DEBUG] 사용할 토픽/서브토픽 데이터:")
    topics = set(item['topic'] for item in json_data)
    subtopics = set(item['sub_topic'] for item in json_data)
    print(f"토픽: {topics}")
    print(f"서브토픽: {subtopics}")
    
    # JSON 데이터 처리
    response = await chatbot.process_json_keywords(user_id, user_input, json_data)
    print(f"챗봇: {response}")
    
    # 연관 네트워크 시각화
    for topic in ["work", "psychological", "life_event"]:
        viz_path = f"test_json_{topic}.png"
        result = await chatbot.visualize_associations(topic, viz_path)
        if result:
            print(f"'{topic}' 연관 네트워크 저장됨: {viz_path}")
    
    # 종료
    await chatbot.shutdown()


if __name__ == "__main__":
    # 실행 모드 선택
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "json_test":
        # JSON 통합 테스트
        asyncio.run(run_json_test())
    else:
        # 인터랙티브 데모 실행
        asyncio.run(run_demo())