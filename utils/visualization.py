# visualization.py 개선 버전
"""
네트워크 시각화 모듈 - 토픽과 서브토픽 구분 개선
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import Optional, Dict
import matplotlib.font_manager as fm
import os
import matplotlib.colors as mcolors

# 한글 폰트 설정 함수는 그대로 유지
def set_korean_font():
    """
    시스템에 설치된 한글 폰트를 찾아 설정
    """
    # 일반 폰트 설정
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 한글 폰트 목록
    korean_fonts = [
        'NanumGothic',
        'Malgun Gothic',
        'Apple SD Gothic Neo',
        'AppleGothic',
        'Noto Sans CJK KR',
        'Gulim',
        'Dotum',
        'Batang'
    ]
    
    # sans-serif에 한글 폰트 추가
    for font_name in korean_fonts:
        try:
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if font_path and not font_path.endswith('DejaVuSans.ttf'):
                # sans-serif 패밀리에 한글 폰트 추가
                plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams.get('font.sans-serif', [])
                print(f"한글 폰트 설정: {font_name}")
                break
        except Exception as e:
            print(f"폰트 {font_name} 설정 실패: {e}")
            continue
    
    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False
    
    return plt.rcParams['font.sans-serif'][0] if plt.rcParams.get('font.sans-serif') else 'sans-serif'


def visualize_association_network(
    graph: nx.DiGraph,
    center_concept: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    title_prefix: str = "연관 네트워크"
) -> Optional[str]:
    """
    연관 네트워크 시각화 - 토픽과 서브토픽 구분 개선 버전
    
    Args:
        graph: NetworkX 그래프
        center_concept: 중심 개념
        save_path: 저장 경로
        show: 화면 표시 여부
        title_prefix: 제목 접두사
    
    Returns:
        저장된 파일 경로 또는 None
    """
    if not graph.has_node(center_concept):
        print(f"경고: '{center_concept}' 개념이 그래프에 없습니다.")
        return None
        
    # 한글 폰트 설정
    font_name = set_korean_font()

    # 피규어 설정 (크기 증가)
    # visualization.py 수정 부분 - 새로운 방법

    # 피규어 크기 자체를 더 키우고 비율 조절
    plt.figure(figsize=(22, 18), dpi=100)  # 높이를 더 키움

    # 제목을 피규어 전체 제목 대신 Axes 내부 제목으로 설정
    plt.suptitle(title, fontsize=24, fontweight='bold', y=0.98)  # suptitle 사용
    # 또는
    plt.gca().set_title(title, fontsize=24, fontweight='bold', pad=30)  # 축 제목 사용

    # 또는 제목을 텍스트로 추가
    plt.text(0.5, 0.97, title, 
            horizontalalignment='center',
            fontsize=24, fontweight='bold',
            transform=plt.gca().transAxes)  # 축 좌표계 사용

    
    # 서브그래프 생성 (중심 개념 기준)
    if center_concept and center_concept in graph:
        # 중심에서 2단계 깊이의 노드만 시각화
        subgraph_nodes = set([center_concept])
        subgraph_nodes.update(graph.neighbors(center_concept))
        
        # 두 번째 단계 이웃도 추가
        second_neighbors = set()
        for node in list(graph.neighbors(center_concept)):
            second_neighbors.update(list(graph.neighbors(node))[:8])  # 최대 8개 이웃
        subgraph_nodes.update(second_neighbors)
        
        # 서브그래프 생성
        subgraph = graph.subgraph(subgraph_nodes)
    else:
        subgraph = graph
    
    # 노드 속성 수집 및 명확한 구분
    node_types = {}
    
    # 모든 노드에 대해 데이터를 검사하고 타입 결정
    for node in subgraph.nodes():
        # 노드 메타데이터에서 타입 정보 확인
        metadata = subgraph.nodes[node].get('metadata', {})
        node_type = metadata.get('type', '')
        
        # 메타데이터에 타입이 없으면 이름 기반으로 추정
        if not node_type:
            if node == center_concept:
                node_type = 'topic'
            elif any(node.startswith(prefix) for prefix in ['topic_', 'subject_']):
                node_type = 'topic'
            elif any(node.startswith(prefix) for prefix in ['sub_', 'subtopic_']):
                node_type = 'subtopic'
            else:
                # 이웃 관계로 추정 (중심 개념의 직접 이웃은 서브토픽으로 간주)
                if center_concept and node in graph.neighbors(center_concept):
                    node_type = 'subtopic'
                else:
                    node_type = 'keyword'
        
        node_types[node] = node_type
    
    # 노드 크기 계산 (토픽 > 서브토픽 > 키워드)
    node_sizes = []
    for node in subgraph.nodes():
        activation_count = subgraph.nodes[node].get('activation_count', 0)
        
        # 노드 유형에 따른 기본 크기 (명확한 구분)
        if node_types.get(node) == 'topic':
            base_size = 2500  # 토픽 크기 증가
        elif node_types.get(node) == 'subtopic':
            base_size = 1500  # 서브토픽 크기 증가
        else:
            base_size = 800   # 키워드도 약간 크게
            
        # 활성화 횟수에 따른 보너스 크기
        size = base_size + min(activation_count * 100, 1000)
        node_sizes.append(size)
    
    # 노드 색상 설정 - 더 뚜렷한 색상 차이
    node_colors = []
    
    # 노드 유형별 색상 (더 명확한 대비)
    type_colors = {
        'topic': '#ff3333',       # 밝은 빨간색 (토픽)
        'subtopic': '#3333ff',    # 밝은 파란색 (서브토픽)
        'keyword': '#33cc33'      # 밝은 녹색 (키워드)
    }
    
    for node in subgraph.nodes():
        node_type = node_types.get(node, 'keyword')
        
        # 중심 노드는 더 강조
        if node == center_concept:
            # 중심 노드가 토픽이면 더 진한 빨강
            if node_type == 'topic':
                node_colors.append('#aa0000')  # 진한 빨강
            else:
                # 중심이 서브토픽이면 진한 파랑
                node_colors.append('#0000aa')  # 진한 파랑
        else:
            # 다른 노드들은 노드 타입에 따른 색상
            node_colors.append(type_colors.get(node_type, '#87CEEB'))
    
    # 엣지 두께 설정 (연결 강도 기반)
    edge_widths = []
    edge_colors = []
    edge_styles = []
    
    # 연결 유형별 스타일
    connection_styles = {
        'hierarchical': 'solid',   # 계층적 관계: 실선
        'semantic': 'dashed',      # 의미적 관계: 점선
        'temporal': 'dashdot',     # 시간적 관계: 대시-점
        'emotional': 'dotted'      # 감정적 관계: 점선
    }
    
    for u, v in subgraph.edges():
        # 연결 강도 기반 두께
        weight = subgraph[u][v].get('weight', 0.5)
        # 비선형 매핑으로 연결 강도 시각적 차이 강화
        edge_widths.append(max(weight * 12, 1.0))  # 선 두께 증가
        
        # 연결 강도 기반 색상 (강한 연결=진한 색, 약한 연결=연한 색)
        if weight >= 0.7:
            # 강한 연결 (진한 색)
            edge_colors.append((0.1, 0.1, 0.1, min(weight * 1.3, 1.0)))
        elif weight >= 0.4:
            # 중간 연결 (중간 색)
            edge_colors.append((0.3, 0.3, 0.3, weight))
        else:
            # 약한 연결 (연한 색)
            edge_colors.append((0.5, 0.5, 0.5, max(weight, 0.3)))
        
        # 연결 유형 기반 스타일
        conn_type = subgraph[u][v].get('type', 'semantic')
        edge_styles.append(connection_styles.get(conn_type, 'solid'))
    
    # 레이아웃 계산 (중심 개념 고정)
    if center_concept and center_concept in subgraph:
        # 스프링 레이아웃 with 중심 고정
        pos = nx.spring_layout(
            subgraph,
            k=1.8/np.sqrt(len(subgraph.nodes())),  # 더 넓게 분포
            iterations=70,  # 반복 증가로 안정성 향상
            seed=42
        )
        # 중심 개념은 중앙에 고정
        pos[center_concept] = np.array([0, 0])
    else:
        # 일반 스프링 레이아웃
        pos = nx.spring_layout(
            subgraph,
            k=1.8/np.sqrt(len(subgraph.nodes())),
            iterations=70
        )
    
    # 연결 유형별로 그리기 (스타일 구분을 위해)
    edge_types = set(data.get('type', 'semantic') for _, _, data in subgraph.edges(data=True))
    
    for edge_type in edge_types:
        # 해당 유형의 엣지만 필터링
        edges_of_type = [(u, v) for u, v in subgraph.edges() if subgraph[u][v].get('type', 'semantic') == edge_type]
        
        if not edges_of_type:
            continue
            
        # 엣지 속성 필터링
        type_widths = [edge_widths[i] for i, (u, v) in enumerate(subgraph.edges()) if subgraph[u][v].get('type', 'semantic') == edge_type]
        type_colors = [edge_colors[i] for i, (u, v) in enumerate(subgraph.edges()) if subgraph[u][v].get('type', 'semantic') == edge_type]
        
        # 선 스타일 결정
        style = connection_styles.get(edge_type, 'solid')
        
        # 엣지 그리기
        nx.draw_networkx_edges(
            subgraph, pos,
            edgelist=edges_of_type,
            width=type_widths,
            edge_color=type_colors,
            style=style,
            arrows=True,
            arrowsize=20,  # 화살표 크기 증가
            arrowstyle='-|>'
        )
    
    # 노드 그리기 (외곽선 추가로 구분 강화)
    for node_type in ['topic', 'subtopic', 'keyword']:
        # 해당 타입의 노드만 필터링
        nodes_of_type = [node for node in subgraph.nodes() if node_types.get(node) == node_type]
        if not nodes_of_type:
            continue
            
        # 타입별 사이즈 추출
        sizes = [node_sizes[i] for i, node in enumerate(subgraph.nodes()) if node_types.get(node) == node_type]
        
        # 타입별 색상 추출
        colors = [node_colors[i] for i, node in enumerate(subgraph.nodes()) if node_types.get(node) == node_type]
        
        # 외곽선 설정
        if node_type == 'topic':
            edgecolors = 'black'  # 토픽은 검정 외곽선
            linewidths = 3.0
        elif node_type == 'subtopic':
            edgecolors = 'darkblue'  # 서브토픽은 파란 외곽선
            linewidths = 2.0
        else:
            edgecolors = 'darkgreen'  # 키워드는 녹색 외곽선
            linewidths = 1.0
        
        # 각 타입별로 노드 그리기
        nx.draw_networkx_nodes(
            subgraph, pos,
            nodelist=nodes_of_type,
            node_size=sizes,
            node_color=colors,
            alpha=0.95,
            edgecolors=edgecolors,
            linewidths=linewidths
        )
    
    # 노드 라벨
    labels = {}
    for node in subgraph.nodes():
        # 유니코드 문자열로 변환하여 저장
        node_type = node_types.get(node)
        
        # 토픽과 서브토픽에는 노드 타입 표시 추가
        if node_type == 'topic':
            labels[node] = f"[토픽] {node}"  # 토픽 라벨 특별 표시
        elif node_type == 'subtopic':
            labels[node] = f"[서브] {node}"  # 서브토픽 라벨 특별 표시
        else:
            labels[node] = str(node)
    
    # 토픽 라벨 먼저 그리기 (더 크게)
    topic_labels = {node: label for node, label in labels.items() if node_types.get(node) == 'topic'}
    nx.draw_networkx_labels(
        subgraph, pos,
        labels=topic_labels,
        font_size=16,  # 토픽은 더 큰 폰트
        font_weight='bold',
        font_family=font_name
    )
    
    # 서브토픽 라벨 그리기
    subtopic_labels = {node: label for node, label in labels.items() if node_types.get(node) == 'subtopic'}
    nx.draw_networkx_labels(
        subgraph, pos,
        labels=subtopic_labels,
        font_size=14,  # 서브토픽은 중간 크기 폰트
        font_weight='bold',
        font_family=font_name
    )
    
    # 키워드 라벨 그리기
    keyword_labels = {node: label for node, label in labels.items() if node_types.get(node) == 'keyword'}
    nx.draw_networkx_labels(
        subgraph, pos,
        labels=keyword_labels,
        font_size=12,  # 키워드는 작은 폰트
        font_family=font_name
    )
    
    # 엣지 가중치 표시 (강한 연결만)
    edge_labels = {}
    for u, v, data in subgraph.edges(data=True):
        weight = data.get('weight', 0)
        if weight > 0.5:  # 강한 연결만 표시 (임계값 낮춤)
            edge_labels[(u, v)] = f"{weight:.2f}"
    
    nx.draw_networkx_edge_labels(
        subgraph, pos,
        edge_labels=edge_labels,
        font_size=11,
        font_family=font_name,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)  # 배경 추가로 가독성 향상
    )
    
    # 범례 추가 - 명확히 구분
    plt.figtext(0.01, 0.01, "범례:", fontsize=14, fontweight='bold')
    
    # 노드 유형 범례
    plt.figtext(0.01, 0.97, "● 토픽 (빨간색)", fontsize=12, color='#ff3333', fontweight='bold')
    plt.figtext(0.20, 0.97, "● 서브토픽 (파란색)", fontsize=12, color='#3333ff', fontweight='bold')
    plt.figtext(0.45, 0.97, "● 키워드 (녹색)", fontsize=12, color='#33cc33')
    
    # 연결 유형 범례
    plt.figtext(0.01, 0.94, "계층적 관계: ――", fontsize=11)
    plt.figtext(0.20, 0.94, "의미적 관계: - - -", fontsize=11)
    plt.figtext(0.40, 0.94, "기타 관계: -·-", fontsize=11)
    
    # 연결 강도 범례
    plt.figtext(0.01, 0.91, "강한 연결 (≥0.7): 굵은 선", fontsize=11)
    plt.figtext(0.25, 0.91, "중간 연결 (0.4~0.7): 중간 선", fontsize=11)
    plt.figtext(0.55, 0.91, "약한 연결 (<0.4): 얇은 선", fontsize=11)
    
    # 타이틀 설정 - 타입 정보 추가
    if center_concept:
        node_type = node_types.get(center_concept, 'unknown')
        type_name = '토픽' if node_type == 'topic' else '서브토픽' if node_type == 'subtopic' else '키워드'
        title = f"{title_prefix}: {type_name} '{center_concept}' 중심"
    else:
        title = f"{title_prefix}"
    
    plt.title(title, fontsize=22, fontweight='bold', pad=30, y=1.05)  # y 위치와 패딩 조정
    
    plt.axis('off')
    plt.tight_layout()
    
    # 배경색 설정 - 약간 엷은 배경으로 가독성 향상
    plt.gca().set_facecolor('#f8f8f8')
    
    # 파일 저장 또는 화면 표시
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return save_path