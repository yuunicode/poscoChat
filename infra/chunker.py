# infra/chunker.py
from config.settings import get_settings 
from config.enums import ChunkingType, ChunkingStrategy
from typing import List, Dict, Any, Optional
import uuid, logging
from bs4 import BeautifulSoup

settings = get_settings()

class Chunker:
    """
    레코드를 청킹하는 구체적인 구현체입니다.
    주어진 chunkers.py의 모든 청킹 로직을 포함합니다.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _create_chunk_record(self
                                 , original_record: Dict[str, Any]
                                 , context: str
                                 , chunk_type: ChunkingStrategy
                                 , **extra_chunk_info
    ) -> Dict[str, Any]:

        """새로운 청크 레코드를 생성하는 헬퍼 함수."""
        # 원본 레코드의 모든 최상위 필드를 얕게 복사합니다.
        new_record = original_record.copy()
        
        # 새로운 ID와 컨텍스트를 할당합니다.
        new_record["id"] = str(uuid.uuid4())
        new_record["context"] = context

        # 메타데이터 딕셔너리를 독립적으로 처리하여 원본 레코드에 영향을 주지 않도록 합니다.
        # 기존 metadata가 있다면 그 내용을 복사하고, 없다면 새로 생성합니다.
        if "metadata" in original_record and isinstance(original_record["metadata"], dict):
            copied_metadata = original_record["metadata"].copy() # metadata 딕셔너리 자체를 복사
            
            # general_info가 있다면 그것도 독립적으로 복사하여 깊이 복사를 수행합니다.
            if "general_info" in copied_metadata and isinstance(copied_metadata["general_info"], dict):
                copied_metadata["general_info"] = copied_metadata["general_info"].copy()
            
            # table_info도 있다면 독립적으로 복사하여 깊이 복사를 수행합니다.
            if "table_info" in copied_metadata and isinstance(copied_metadata["table_info"], dict):
                copied_metadata["table_info"] = copied_metadata["table_info"].copy()
            
            new_record["metadata"] = copied_metadata
        else:
            new_record["metadata"] = {}

        # chunk_info를 metadata 내에 새로운 필드로 추가합니다.
        new_record["metadata"]["chunk_info"] = {
            "chunk_type": chunk_type.value,
            "original_id": original_record.get("id"),
            **extra_chunk_info
        }
        return new_record


    # -----------------------------------------------------------------------------
    # Narrative Chunking Helper Functions
    # -----------------------------------------------------------------------------

    def _fixed_chunk(self, records: List[Dict[str, Any]], chunk_size: int = 500, **kwargs) -> List[Dict[str, Any]]:
        """ 고정길이 청킹 """
        self.logger.info(f"고정 길이 청킹을 적용합니다. 크기 = {chunk_size}.")
        
        chunked_records = []
        for rec in records:
            text = rec["context"]
            if len(text) <= chunk_size:
                chunked_records.append(rec)
                continue

            for i in range(0, len(text), chunk_size):
                chunk_text = text[i:i + chunk_size]
                new_rec = self._create_chunk_record(
                    original_record = rec,
                    context         = chunk_text,
                    chunk_type      = ChunkingStrategy.NARRATIVE,
                    start_index     = i,
                    end_index       = i + len(chunk_text)
                )
                chunked_records.append(new_rec)
        
        self.logger.info(f" 고정 길이 청킹 완료. 크기: {len(chunked_records)}")
        return chunked_records

    def _recursive_chunk(self, records: List[Dict[str, Any]], chunk_size: int = 500, chunk_overlap: int = 50, **kwargs) -> List[Dict[str, Any]]:
        """ 재귀 기법 청킹 """
        self.logger.info(f" 재귀 청킹을 적용합니다. 크기 = {chunk_size}, 겹침 크기 = {chunk_overlap}.")
        
        chunked_records = []
        for rec in records:
            text = rec["context"]
            if len(text) <= chunk_size:
                chunked_records.append(rec)
                continue
            
            # Simplified recursive splitting logic.
            temp_chunks = []
            current_index = 0
            while current_index < len(text):
                end_index = min(current_index + chunk_size, len(text))
                chunk_text = text[current_index:end_index]
                temp_chunks.append(chunk_text)
                current_index += (chunk_size - chunk_overlap)
                if current_index >= len(text) and (len(text) - (current_index - (chunk_size - chunk_overlap))) < chunk_overlap: # Ensure last chunk is not too small and prevent infinite loop
                    break
            
            for chunk_text in temp_chunks:
                new_rec = self._create_chunk_record(
                    original_record = rec,
                    context         = chunk_text,
                    chunk_type      = ChunkingStrategy.RECURSIVE,
                    chunk_overlap   = chunk_overlap
                )
                chunked_records.append(new_rec)
        
        self.logger.info(f" 재귀 청킹 완료. 크기: {len(chunked_records)}")
        return chunked_records

    def _hierarchy_chunk(self, records: List[Dict[str, Any]], max_depth: int = None, **kwargs) -> List[Dict[str, Any]]:
        """
        `section_path` 메타데이터를 사용하여 레코드를 계층 구조에 따라 묶어 청킹합니다.
        `max_depth` 인자를 통해 특정 깊이까지만 묶고 그 하위는 분리된 청크로 생성할 수 있습니다.

        예를 들어, max_depth = 1이면 '1장' 레벨에서만 묶고 '1.1절'부터는 새로운 청크가 됩니다.
        max_depth가 None이거나 더 깊으면, 가능한 한 가장 깊은 계층까지 묶으려 시도합니다.

        Args:
            records (List[Dict[str, Any]]): 청킹할 레코드 목록.
            max_depth (int, optional): 청크를 묶을 섹션 계층 구조의 최대 깊이.
                                        None이면 모든 깊이를 고려하여 동일한 계층 경로를 가진 요소를 묶습니다.
                                        0부터 시작하는 인덱스(예: 0은 최상위 제목, 1은 다음 레벨).
            **kwargs: `_create_chunk_record` 함수에 전달할 추가 인자.

        Returns:
            List[Dict[str, Any]]: 계층 청킹이 적용된 레코드 목록.
        """
        self.logger.info(f" 계층 청킹을 적용합니다. 최대 깊이 = {max_depth if max_depth is not None else '제한 없음'}.")
        chunked_records = []
        current_hierarchy_chunk_recs = []
        current_hierarchy_path_for_chunk = None

        for rec in records:
            # general_info에서 sectionX 필드를 사용하여 계층 구조를 리스트로 재구성
            general_info = rec.get("metadata", {}).get("general_info", {})
            full_section_hierarchy_list = []
            for i in range(5): 
                section_key = f"section{i}"
                if general_info.get(section_key):
                    full_section_hierarchy_list.append(general_info[section_key])
                else:
                    break

            # `comparison_path`는 _hierarchy_chunk의 `max_depth` 파라미터에 따라 결정
            # 이 파라미터는 계층 묶음의 "깊이"를 제어합니다.
            comparison_path = full_section_hierarchy_list[:max_depth] if max_depth is not None else full_section_hierarchy_list

            # 계층 경로가 없거나(제목 카테고리가 아닌 경우), 현재 묶음에 추가할 수 없는 경우
            if not comparison_path and rec.get("category") != "Title":
                if current_hierarchy_chunk_recs:
                    combined_context = "\n\n".join([r["context"] for r in current_hierarchy_chunk_recs])
                    first_rec_in_chunk = current_hierarchy_chunk_recs[0]
                    # _create_chunk_record에 전달할 hierarchy_path 재구성
                    general_info_for_chunk = first_rec_in_chunk.get("metadata", {}).get("general_info", {})
                    full_section_hierarchy_list_for_chunk = []
                    for i in range(5):
                        section_key = f"section{i}"
                        if general_info_for_chunk.get(section_key):
                            full_section_hierarchy_list_for_chunk.append(general_info_for_chunk[section_key])
                        else:
                            break
                    new_rec = self._create_chunk_record(
                        original_record         = first_rec_in_chunk,
                        context                 = combined_context,
                        chunk_type              = ChunkingStrategy.HIERARCHICAL,
                        hierarchy_path          = " -> ".join(full_section_hierarchy_list_for_chunk),
                        contained_original_ids  = [r["id"] for r in current_hierarchy_chunk_recs]
                    )
                    chunked_records.append(new_rec)
                    current_hierarchy_chunk_recs = []
                    current_hierarchy_path_for_chunk = None
                chunked_records.append(rec) # 계층 묶음에 포함되지 않는 레코드는 그대로 추가
                continue

            if not current_hierarchy_chunk_recs:
                current_hierarchy_chunk_recs.append(rec)
                current_hierarchy_path_for_chunk = comparison_path
            elif comparison_path == current_hierarchy_path_for_chunk:
                current_hierarchy_chunk_recs.append(rec)
            else:
                combined_context = "\n\n".join([r["context"] for r in current_hierarchy_chunk_recs])
                first_rec_in_chunk = current_hierarchy_chunk_recs[0]
                
                # _create_chunk_record에 전달할 hierarchy_path 재구성
                general_info_for_chunk = first_rec_in_chunk.get("metadata", {}).get("general_info", {})
                full_section_hierarchy_list_for_chunk = []
                for i in range(5):
                    section_key = f"section{i}"
                    if general_info_for_chunk.get(section_key):
                        full_section_hierarchy_list_for_chunk.append(general_info_for_chunk[section_key])
                    else:
                        break

                new_rec = self._create_chunk_record(
                    original_record         = first_rec_in_chunk,
                    context                 = combined_context,
                    chunk_type              = ChunkingStrategy.HIERARCHICAL,
                    hierarchy_path          = " -> ".join(full_section_hierarchy_list_for_chunk),
                    contained_original_ids  = [r["id"] for r in current_hierarchy_chunk_recs],
                    hierarchy_chunk_depth   = max_depth
                )
                chunked_records.append(new_rec)
                
                current_hierarchy_chunk_recs = [rec]
                current_hierarchy_path_for_chunk = comparison_path
        
        # 루프 종료 후 남아있는 청크 처리
        if current_hierarchy_chunk_recs:
            combined_context = "\n\n".join([r["context"] for r in current_hierarchy_chunk_recs])
            first_rec_in_chunk = current_hierarchy_chunk_recs[0]
            
            # _create_chunk_record에 전달할 hierarchy_path 재구성
            general_info_for_chunk = first_rec_in_chunk.get("metadata", {}).get("general_info", {})
            full_section_hierarchy_list_for_chunk = []
            for i in range(5):
                section_key = f"section{i}"
                if general_info_for_chunk.get(section_key):
                    full_section_hierarchy_list_for_chunk.append(general_info_for_chunk[section_key])
                else:
                    break

            new_rec = self._create_chunk_record(
                original_record         = first_rec_in_chunk,
                context                 = combined_context,
                chunk_type              = ChunkingStrategy.HIERARCHICAL,
                hierarchy_path          = " -> ".join(full_section_hierarchy_list_for_chunk),
                contained_original_ids  = [r["id"] for r in current_hierarchy_chunk_recs],
                hierarchy_chunk_depth   = max_depth
            )
            chunked_records.append(new_rec)

        self.logger.info(f" 계층 청킹 완료. 크기: {len(chunked_records)}")
        return chunked_records
    
    # -----------------------------------------------------------------------------
    # Table Chunking Helper Functions (클래스 메서드로 변경)
    # -----------------------------------------------------------------------------

    def _table_row_chunk(self, records: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Chunks table records by individual rows.
        """
        self.logger.info(" 테이블 행기반 청킹을 적용합니다.")
        chunked_records = []

        for rec in records:
            table_info = rec["metadata"].get("table_info", {})
            html_content = table_info.get("text_as_html")
            
            if not html_content:
                chunked_records.append(rec)
                continue

            soup = BeautifulSoup(html_content, "html.parser")
            table = soup.find("table")
            if not table:
                chunked_records.append(rec)
                continue
            
            rows = table.find_all("tr")
            if not rows:
                chunked_records.append(rec)
                continue

            header_row_text = ""
            if len(rows) > 0:
                header_cells = rows[0].find_all(["th", "td"])
                header_row_text = " | ".join([cell.get_text(strip=True) for cell in header_cells])

            for i, row in enumerate(rows):
                row_content_list = [cell.get_text(strip=True) for cell in row.find_all(["th", "td"])]
                row_content = " | ".join(row_content_list)
                
                full_chunk_context = f"Table Title: {table_info.get('table_title', 'N/A')}\n"
                full_chunk_context += f"Table ID: {table_info.get('table_id', 'N/A')}\n"
                full_chunk_context += f"Row {i+1} content:\n"
                if header_row_text:
                    full_chunk_context += f"Headers: {header_row_text}\n"
                full_chunk_context += row_content

                new_rec = self._create_chunk_record(
                    original_record             = rec,
                    context                     = full_chunk_context,
                    chunk_type                  = ChunkingStrategy.ROW,
                    table_id                    = table_info.get("table_id"),
                    row_index                   = i,
                    original_table_context_id   = rec["id"] 
                )
                chunked_records.append(new_rec)
        
        self.logger.info(f" 행기반 청킹 완료. 총 크기: {len(chunked_records)}")
        return chunked_records

    # def _table_col_chunk(self, records: List[Dict[str, Any]], columns_per_chunk: int = 3, **kwargs) -> List[Dict[str, Any]]:
    #     """
    #     Chunks table records by columns or groups of columns.
    #     """
    #     self.logger.info(f" 테이블 열기반 청킹을 적용합니다. (columns_per_chunk={columns_per_chunk}).")

    #     chunked_records = []
    #     for rec in records:
    #         table_info = rec["metadata"].get("table_info", {})
    #         html_content = table_info.get("text_as_html")
            
    #         if not html_content:
    #             chunked_records.append(rec)
    #             continue

    #         soup = BeautifulSoup(html_content, "html.parser")
    #         table = soup.find("table")
    #         if not table:
    #             chunked_records.append(rec)
    #             continue
            
    #         rows = table.find_all("tr")
    #         if not rows:
    #             chunked_records.append(rec)
    #             continue

    #         table_data = []
    #         for row in rows:
    #             table_data.append([cell.get_text(strip = True) for cell in row.find_all(["th", "td"])])
            
    #         if not table_data:
    #             chunked_records.append(rec)
    #             continue

    #         header_row = table_data[0] if table_data else []
    #         num_cols = len(header_row)

    #         for col_start_idx in range(0, num_cols, columns_per_chunk):
    #             col_end_idx = min(col_start_idx + columns_per_chunk, num_cols)
                
    #             chunk_columns_data = []
    #             for row_data in table_data:
    #                 chunk_columns_data.append(row_data[col_start_idx:col_end_idx])
                
    #             col_chunk_context = f"Table Title: {table_info.get('table_title', 'N/A')}\n"
    #             col_chunk_context += f"Table ID: {table_info.get('table_id', 'N/A')}\n"
    #             col_chunk_context += f"Columns {col_start_idx+1}-{col_end_idx} content:\n"
                
    #             if header_row:
    #                 col_chunk_context += " | ".join(header_row[col_start_idx:col_end_idx]) + "\n"
                
    #             for row_data in chunk_columns_data[1:]:
    #                 col_chunk_context += " | ".join(row_data) + "\n"

    #             new_rec = self._create_chunk_record(
    #                 original_record             = rec,
    #                 context                     = col_chunk_context.strip(),
    #                 chunk_type                  = ChunkingStrategy.COL,
    #                 table_id                    = table_info.get("table_id"),
    #                 column_indices              = (col_start_idx, col_end_idx - 1),
    #                 original_table_context_id   = rec["id"]
    #             )
    #             chunked_records.append(new_rec)
        
    #     self.logger.info(f" 테이블 열기반 청킹 완료. 총 크기: {len(chunked_records)}")
    #     return chunked_records

    
    def apply_text_chunking(self
                            , records: List[Dict[str, Any]]
                            , strategy: ChunkingStrategy
                            , **kwargs
    ) -> List[Dict[str, Any]]:
        """
        텍스트 레코드에 지정된 청킹 전략을 적용합니다.

        Args:
            records (List[Dict[str, Any]]): 청킹할 텍스트 레코드들의 목록입니다.
            strategy (ChunkingStrategy): 텍스트 청킹에 사용될 특정 전략입니다 (예: 'fixed', 'recursive', 'hierarchy').
            **kwargs: 청킹 전략에 특화된 추가 인자들입니다 (예: chunk_size).

        Returns:
            List[Dict[str, Any]]: 청킹이 적용된 텍스트 레코드 목록.
        """
 
        processed_records: List[Dict[str, Any]] = []
        if strategy == ChunkingStrategy.FIXED:
            processed_records = self._fixed_chunk(records, **kwargs)
        elif strategy == ChunkingStrategy.RECURSIVE:
            processed_records = self._recursive_chunk(records, **kwargs)
        elif strategy == ChunkingStrategy.HIERARCHICAL:
            processed_records = self._hierarchy_chunk(records, **kwargs)
        else:
            self.logger.error(f" 유효하지 않은 텍스트 청킹 전략입니다: '{strategy}'. 원본 텍스트 레코드를 반환합니다.")
            return records
        
        return processed_records

    def apply_table_chunking(self
                            , records: List[Dict[str, Any]]
                            , strategy: ChunkingStrategy
                            , **kwargs
    ) -> List[Dict[str, Any]]:
        """
        테이블 레코드에 지정된 청킹 전략을 적용합니다.

        Args:
            records (List[Dict[str, Any]]): 청킹할 테이블 레코드들의 목록입니다.
            strategy (ChunkingStrategy): 테이블 청킹에 사용될 특정 전략입니다 (예: 'row', 'col').
            **kwargs: 청킹 전략에 특화된 추가 인자들입니다 (예: columns_per_chunk).

        Returns:
            List[Dict[str, Any]]: 청킹이 적용된 테이블 레코드 목록.
        """

        processed_records: List[Dict[str, Any]] = []
        if strategy == ChunkingStrategy.ROW:
            processed_records = self._table_row_chunk(records, **kwargs)
        # elif strategy == ChunkingStrategy.COL:
        #     processed_records = self._table_col_chunk(records, **kwargs)
        else:
            self.logger.error(f" 유효하지 않은 테이블 청킹 전략입니다: '{strategy}'. 원본 테이블 레코드를 반환합니다.")
            return records
        
        
        return processed_records

    
    # -----------------------------------------------------------------------------
    # Main Entry Point for Chunking
    # -----------------------------------------------------------------------------
    def apply_chunking(self
                       , records: List[Dict[str, Any]]
                       , chunking_type: ChunkingType
                       , strategy: Optional[ChunkingStrategy] = None
                       , **kwargs
    ) -> List[Dict[str, Any]]:

        """
        records에 지정된 청킹 전략을 사용합니다.

        Args:
            records (List[Dict[str, Any]]): 청킹할 레코드들의 목록입니다.
            chunking_type (ChunkingType): 청킹의 일반적인 유형입니다 (예: 'narrative' (서술형), 'table' (테이블)).
            strategy (ChunkingStrategy, 선택 사항): 일반적인 유형 내에서 사용될 특정 전략입니다.
            **kwargs: 청킹 전략에 특화된 추가 인자들입니다 (예: chunk_size, columns_per_chunk).

        Returns:
            List[Dict[str, Any]]: 청킹된 레코드 목록.
        """
        
        text_records = [rec for rec in records if rec.get("category") != "Table"]
        table_records = [rec for rec in records if rec.get("category") == "Table"]

        processed_records: List[Dict[str, Any]] = []

        if chunking_type == ChunkingType.NARRATIVE:
            processed_text_records = self.apply_text_chunking(text_records, strategy, **kwargs)
            processed_records.extend(processed_text_records)
            processed_records.extend(table_records) # 테이블 레코드는 변경 없이 그대로 추가
        
        elif chunking_type == ChunkingType.TABLE:
            processed_table_records = self.apply_table_chunking(table_records, strategy, **kwargs)
            processed_records.extend(processed_table_records)
            processed_records.extend(text_records) # 텍스트 레코드는 변경 없이 그대로 추가
        
        else:
            self.logger.warning(f" 조건에 없는 ChunkingType: '{chunking_type}'. 청킹을 적용하지 않습니다.")
            return records
        
        return processed_records