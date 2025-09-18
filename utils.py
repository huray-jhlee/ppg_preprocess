import numpy as np
from zoneinfo import ZoneInfo
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple

def to3(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        vv = list(v)
        return [vv[0] if len(vv)>0 else np.nan,
                vv[1] if len(vv)>1 else np.nan,
                vv[2] if len(vv)>2 else np.nan]
    return [np.nan, np.nan, np.nan]

def convert_date(date_str: str) -> str:
    year = "20" + date_str[:2]
    month = date_str[2:4]
    day = date_str[4:6]
    return f"{year}-{month}-{day}"

def label_filtering(label_datas, target_date_str):
    
    target_har_labels = {}
    
    # for label_data in tqdm(label_datas, desc="select target date in labeling"):
    for label_data in label_datas:
        date = label_data["timeString"].split(" ")[0]
        if date == target_date_str:
            timestamp = label_data["timeStamp"]
            dt_timestamp = datetime.fromtimestamp(timestamp/1000.0, tz=ZoneInfo("Asia/Seoul"))
            
            label_dict = label_data.copy()
            for del_key in ["timeStamp", "timeString"]:
                label_dict.pop(del_key)
            target_har_labels[dt_timestamp] = label_dict
            
    return target_har_labels

def extract_behavior_ranges(
    tag_dict: Dict[datetime, Dict[str, bool]]
) -> Dict[str, List[Tuple[datetime, datetime]]]:
    """
    주어진 시간별 행동 태그(tag_dict)를 기반으로 각 행동의 (시작, 종료) 구간 추출
    - 병합 행동(meal, alcohol): 함께 True인 동안은 동일한 시작 시점에서 구간 기록
    - 단독 행동(cigarette, htp, vape): 독립적으로 True/False 기준으로 구간 기록

    Args:
        tag_dict (Dict[datetime, Dict[str, bool]]): 시간순 행동 상태 변화 기록

    Returns:
        Dict[str, List[Tuple[datetime, datetime]]]: 각 행동별 (start, end) 범위 리스트
    """
    
    behaviors = ["meal", "cigarette", "htp", "vape", "alcohol"]
    merge_behaviors = {"meal", "alcohol"}
    solo_behaviors = {"cigarette", "htp", "vape"}
    
    # 초기 상태
    prev_state = {b: False for b in behaviors}
    ongoing_ranges = {}  # 행동명 -> 시작시간
    result = defaultdict(list)
    
    # 시간 순 정렬
    sorted_items = sorted(tag_dict.items())
    
    for i, (ts, curr_state) in enumerate(sorted_items):
        # 병합 행동 처리
        prev_merge_active = {b for b in merge_behaviors if prev_state[b]}
        curr_merge_active = {b for b in merge_behaviors if curr_state[b]}
        
        # 병합 종료 조건: 기존 켜짐 집합 ⊄ 현재 켜짐 집합
        if prev_merge_active and not prev_merge_active.issubset(curr_merge_active):
            start_time = ongoing_ranges.get("__merge__")
            end_time = ts
            if start_time:
                for b in prev_merge_active:
                    result[b].append((start_time, end_time))
                ongoing_ranges.pop("__merge__", None)
                for b in prev_merge_active:
                    ongoing_ranges.pop(b, None)
        
        # 병합 시작 조건: 새롭게 병합 구간 진입
        if curr_merge_active and not prev_merge_active:
            # 이미 시작된 행동이 있으면 그 중 가장 과거 시점을 병합 시작 시점으로 사용
            existsing_starts = [ongoing_ranges.get(b) for b in curr_merge_active if b in ongoing_ranges]
            if existsing_starts:
                start_time = min(existsing_starts)
            else :
                start_time = ts

            ongoing_ranges["__merge__"] = start_time
            for b in curr_merge_active:
                ongoing_ranges[b] = start_time
        
        # 단독 행동 처리
        for b in solo_behaviors:
            prev = prev_state[b]
            curr = curr_state[b]
            if not prev and curr:
                ongoing_ranges[b] = ts
            elif prev and not curr:
                start_time = ongoing_ranges.pop(b, None)
                if start_time:
                    result[b].append((start_time, ts))
        
        # 상태 갱신
        prev_state = curr_state.copy()
    
    # 마지막 범위 정리
    if sorted_items:
        last_ts = sorted_items[-1][0]
        
        # 병합 그룹 종료 처리
        merge_active = {b for b in merge_behaviors if prev_state[b]}
        if merge_active and "__merge__" in ongoing_ranges:
            start_time = ongoing_ranges.pop("__merge__")
            for b in merge_active:
                result[b].append((start_time, last_ts))
                ongoing_ranges.pop(b, None)
        
        # 단독 행동 종료 처리
        for b in solo_behaviors:
            if prev_state[b] and b in ongoing_ranges:
                start_time = ongoing_ranges.pop(b)
                result[b].append((start_time, last_ts))
    
    return dict(result)

def label_window_from_ranges(
    window_start: datetime,
    window_end: datetime,
    behavior_ranges: Dict[str, List[Tuple[datetime, datetime]]],
    behavior_order: List[str] = ["meal", "cigarette", "htp", "vape", "alcohol"]
) -> List[int]:
    labels = []
    for behavior in behavior_order:
        has_overlap = any(
            (start < window_end and end > window_start)
            for (start, end) in behavior_ranges.get(behavior, [])
        )
        labels.append(1 if has_overlap else 0)
    return labels