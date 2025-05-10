"""
SQLite 저장소 클래스
단기 메모리 및 관계 데이터 관리
"""
import sqlite3
import json
from datetime import datetime
from typing import List, Optional, Dict, Any

from models.memory_entry import MemoryEntry
from models.enums import MemoryTier


class SQLiteStorage:
    """
    SQLite 단기 메모리 저장소
    키워드 중심 연결 강도 관리
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 메모리 테이블 생성
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    concepts TEXT NOT NULL,
                    importance REAL DEFAULT 0.5,
                    emotional_weight REAL DEFAULT 0.0,
                    access_count INTEGER DEFAULT 0,
                    tier TEXT NOT NULL,
                    metadata TEXT,
                    last_accessed TIMESTAMP,
                    creation_time TIMESTAMP NOT NULL
                )
            ''')
            
            # 개념 테이블 생성
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS concepts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    concept TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES memories(id)
                )
            ''')
            
            # 키워드 간 연결 강도를 관리하는 테이블 추가
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS concept_connections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_concept TEXT NOT NULL,
                    target_concept TEXT NOT NULL,
                    weight REAL DEFAULT 0.5,
                    connection_type TEXT DEFAULT 'semantic',
                    strengthening_count INTEGER DEFAULT 1,
                    last_updated TIMESTAMP NOT NULL,
                    creation_time TIMESTAMP NOT NULL,
                    UNIQUE(source_concept, target_concept)
                )
            ''')
            
            # 인덱스 생성
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_concepts ON memories(concepts)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_creation_time ON memories(creation_time)')

            # 개념 테이블 인덱스
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_concept_text ON concepts(concept)')
            
            # 연결 테이블 인덱스
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_source_concept ON concept_connections(source_concept)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_target_concept ON concept_connections(target_concept)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_connection_weight ON concept_connections(weight)')

            conn.commit()
                
    
    async def save(self, memory: MemoryEntry) -> None:
        """메모리 저장"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO memories 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                memory.id,
                json.dumps(memory.content),
                json.dumps(memory.concepts),
                memory.importance,
                memory.emotional_weight,
                memory.access_count,
                memory.tier.value,
                json.dumps(memory.metadata),
                memory.last_accessed.isoformat() if memory.last_accessed else None,
                memory.creation_time.isoformat()
            ))
            
            # 개념 테이블에 저장
            for concept in memory.concepts:
                cursor.execute('''
                    INSERT INTO concepts (concept, memory_id)
                    VALUES (?, ?)
                ''', (concept, memory.id))
            
            conn.commit()
    
    async def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """메모리 조회"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM memories WHERE id = ?', (memory_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_memory(row)
            return None
    
    async def find_by_concepts(self, concepts: List[str], limit: int = 10) -> List[MemoryEntry]:
        """개념으로 메모리 검색"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 간단한 LIKE 쿼리로 변경
            where_clauses = []
            params = []
            
            for concept in concepts:
                where_clauses.append("concepts LIKE ?")
                params.append(f'%{concept}%')
            
            if not where_clauses:
                return []
            
            query = f'''
                SELECT * FROM memories 
                WHERE {" OR ".join(where_clauses)}
                ORDER BY importance DESC, last_accessed DESC
                LIMIT ?
            '''
            
            params.append(limit)
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [self._row_to_memory(row) for row in rows]
    
    async def update_access(self, memory_id: str) -> None:
        """접근 정보 업데이트"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE memories 
                SET access_count = access_count + 1,
                    last_accessed = ? 
                WHERE id = ?
            ''', (datetime.now().isoformat(), memory_id))
            
            conn.commit()
    
    async def find_by_tier(self, tier: MemoryTier) -> List[MemoryEntry]:
        """티어별 메모리 조회"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM memories WHERE tier = ?', (tier.value,))
            rows = cursor.fetchall()
            
            return [self._row_to_memory(row) for row in rows]
    
    async def delete(self, memory_id: str) -> None:
        """메모리 삭제"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 먼저 관련 개념 삭제
            cursor.execute('DELETE FROM concepts WHERE memory_id = ?', (memory_id,))
            
            # 메모리 삭제
            cursor.execute('DELETE FROM memories WHERE id = ?', (memory_id,))
            
            conn.commit()
    
    def _row_to_memory(self, row: tuple) -> MemoryEntry:
        """데이터베이스 행을 MemoryEntry로 변환"""
        return MemoryEntry(
            id=row[0],
            content=json.loads(row[1]),
            concepts=json.loads(row[2]),
            importance=row[3],
            emotional_weight=row[4],
            access_count=row[5],
            tier=MemoryTier(row[6]),
            metadata=json.loads(row[7]) if row[7] else {},
            last_accessed=datetime.fromisoformat(row[8]) if row[8] else None,
            creation_time=datetime.fromisoformat(row[9])
        )
    
    # 키워드 연결 관련 메서드 추가
    async def save_concept_connection(
        self, 
        source_concept: str, 
        target_concept: str, 
        weight: float = 0.5,
        connection_type: str = 'semantic'
    ) -> None:
        """키워드 간 연결 저장"""
        current_time = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 기존 연결 확인
            cursor.execute('''
                SELECT weight, strengthening_count FROM concept_connections
                WHERE source_concept = ? AND target_concept = ?
            ''', (source_concept, target_concept))
            
            existing = cursor.fetchone()
            
            if existing:
                # 기존 연결 강화
                old_weight, strengthening_count = existing
                new_weight = min(1.0, old_weight + weight * 0.2)
                new_count = strengthening_count + 1
                
                cursor.execute('''
                    UPDATE concept_connections
                    SET weight = ?,
                        strengthening_count = ?,
                        last_updated = ?
                    WHERE source_concept = ? AND target_concept = ?
                ''', (new_weight, new_count, current_time, source_concept, target_concept))
            else:
                # 새 연결 생성
                cursor.execute('''
                    INSERT INTO concept_connections
                    (source_concept, target_concept, weight, connection_type, last_updated, creation_time)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (source_concept, target_concept, weight, connection_type, current_time, current_time))
            
            conn.commit()
    
    async def get_concept_connections(self, concept: str, min_weight: float = 0.1) -> List[Dict[str, Any]]:
        """특정 개념과 연결된 개념들 조회"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 양방향 연결 조회
            cursor.execute('''
                SELECT source_concept, target_concept, weight, connection_type, strengthening_count
                FROM concept_connections
                WHERE (source_concept = ? OR target_concept = ?) AND weight >= ?
                ORDER BY weight DESC
            ''', (concept, concept, min_weight))
            
            rows = cursor.fetchall()
            
            # 결과 가공
            connections = []
            for row in rows:
                source, target, weight, conn_type, count = row
                
                # 방향 일관성 유지 (조회한 개념은 항상 source로)
                if target == concept:
                    source, target = target, source
                
                connections.append({
                    'source': source,
                    'target': target,
                    'weight': weight,
                    'type': conn_type,
                    'count': count
                })
            
            return connections
    
    async def apply_connection_decay(self, decay_factor: float = 0.95) -> None:
        """연결 강도 감소 적용"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 모든 연결 강도 감소
            cursor.execute('''
                UPDATE concept_connections
                SET weight = weight * ?
            ''', (decay_factor,))
            
            # 최소 강도 이하의 연결 삭제
            cursor.execute('''
                DELETE FROM concept_connections
                WHERE weight < 0.1
            ''')
            
            conn.commit()
    
    def get_stats(self) -> Dict[str, Any]:
        """저장소 통계"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 총 메모리 수
            cursor.execute('SELECT COUNT(*) FROM memories')
            total_count = cursor.fetchone()[0]
            
            # 티어별 분포
            cursor.execute('''
                SELECT tier, COUNT(*) 
                FROM memories 
                GROUP BY tier
            ''')
            tier_distribution = dict(cursor.fetchall())
            
            # 평균 중요도
            cursor.execute('SELECT AVG(importance) FROM memories')
            avg_importance = cursor.fetchone()[0] or 0.0
            
            # 키워드 연결 통계
            cursor.execute('SELECT COUNT(*) FROM concept_connections')
            connection_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT AVG(weight) FROM concept_connections')
            avg_connection_weight = cursor.fetchone()[0] or 0.0
            
            return {
                'total_memories': total_count,
                'tier_distribution': tier_distribution,
                'average_importance': avg_importance,
                'connection_stats': {
                    'total_connections': connection_count,
                    'average_weight': avg_connection_weight
                }
            }
    async def get_weak_connections(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        지정된 임계값보다 약한 연결들을 조회
        
        Args:
            threshold: 연결 강도 임계값 (이 값 이하의 연결이 조회됨)
        
        Returns:
            약한 연결 목록 (소스 개념, 타겟 개념, 연결 강도)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    source_concept, 
                    target_concept, 
                    weight, 
                    connection_type, 
                    last_updated
                FROM concept_connections
                WHERE weight <= ?
                ORDER BY weight ASC
            ''', (threshold,))
            
            rows = cursor.fetchall()
            
            # 결과 가공
            weak_connections = []
            for row in rows:
                source, target, weight, conn_type, last_updated = row
                weak_connections.append({
                    'source': source,
                    'target': target,
                    'weight': weight,
                    'type': conn_type,
                    'last_updated': last_updated
                })
            
            return weak_connections