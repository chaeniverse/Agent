"""
meeting_agent.py

채님의 Obsidian vault에 회의록을 자동 정리해 저장하는 Claude Agent.

사용법:
    pip install claude-agent-sdk
    python meeting_agent.py <transcript_path>

예:
    python meeting_agent.py ~/Downloads/2026-04-21_lpl_raw.md
    # -> "/Users/chaehyun/.../01. PIPET/2026-04-21 lpl 회의록.md" 생성

환경변수로 저장 경로 바꾸기:
    export MEETING_NOTES_DIR="/path/to/other/vault"
"""

import asyncio
import os
import sys
from pathlib import Path

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
)

# ------------------------------------------------------------
# 설정: Obsidian vault 저장 경로
# ------------------------------------------------------------
OUTPUT_DIR = Path(
    os.environ.get(
        "MEETING_NOTES_DIR",
        "/Users/chaehyun/Library/CloudStorage/Dropbox/Obsidian Vault/01. PIPET",
    )
)

# ------------------------------------------------------------
# 시스템 프롬프트: 채님의 실제 회의록 4건에서 추출한 포맷
# ------------------------------------------------------------
SYSTEM_PROMPT = f"""당신은 채님(임상 연구 협업)의 회의록 정리 전문가입니다.
입력된 회의 원문(거친 메모, 녹취록, 혹은 일부 정리된 노트)을 읽고,
채님의 Obsidian vault 포맷에 맞춰 정리한 후 파일로 저장합니다.

# 저장 경로 (절대경로)
{OUTPUT_DIR}

# 파일명 규칙
`YYYY-MM-DD <주제> 회의록.md`
- 날짜: 원문 YAML frontmatter → 본문 언급 → 입력 파일명 순으로 찾음.
- 주제: 짧은 라벨. 실제 사용 예시:
    - `lpl`           (LPL 연구)
    - `PIPET 마약`    (multiple myeloma opioid 연구)
- 확장자는 반드시 `.md`.

# 출력 포맷

## 1) YAML frontmatter (파일 맨 위, 반드시 포함)
---
project: PIPET
tags:
  - <영역/세부주제>
type: 회의록
date: YYYY-MM-DD
---

실제 tag 예시:
- 혈액내과/lpl
- 혈액내과/마약데이터분석

원문에 frontmatter가 이미 있으면 그대로 유지하고 비어있는 필드만 보완.

## 2) 본문 작성 원칙
- 기본 단위는 **bullet (`- `)**. 반응·부연·예시는 서브 bullet (들여쓰기).
- 원문에 **발화 인용이 있으면**:
    1) `>` blockquote로 그대로 인용 (여러 줄 가능)
    2) 바로 아래 `**요지**` 헤더 + 1~3개 bullet로 압축 요약
  (2026-04-07 회의록 스타일)
- 주제가 여러 개로 나뉘면 **짧은 소제목 한 줄**로 구분.
  예: `컷오프 및 서브그룹`, `Low Risk 결과 해석`, `변수 축소 제안`
  (과도한 `#`, `##` 남발 금지. 흐름이 자연스러우면 소제목 없어도 됨)
- 한국어 + 영어 + 의학 용어가 섞인 원문은 **그대로 유지**. 번역 금지.
  (예: bortezomib, TLT12, multivariable, Youden index, AIC 등은 영어 그대로)
- 구어체의 말끝 흐림 ("~ 인 듯", "~ 인 거죠", "~ 이지 않을까") 은 보존 가능.
  의미 없는 추임새, 중복, 잡담만 정리.
- 원문에 없는 내용은 추측하지 않는다.
  불명확하면 `(불명확)`, 확인이 필요하면 `(확인 필요)` 로 표기.

## 3) 끝부분 특수 섹션 (원문에 해당 요소가 있을 때만)
본문 뒤에 `---` 구분선을 두고 이어 붙임.

### Action items (할 일이 명시되어 있을 때)
---
💡Action items
[] 할 일 1
[] 할 일 2

### 알짜배기 (재사용 가능한 결정/기준/가이드가 풍부할 때)
---

알짜배기

[카테고리1 예: Baseline 기술 참고]
- ...

[카테고리2 예: 분석 관련]
- ...

# 작업 순서 (반드시 이 순서로)
1. `Read` 도구로 입력 파일 전체를 읽는다.
2. 위 포맷에 맞춰 정리된 마크다운을 머리 속에서 완성한다.
3. `Write` 도구로 `{OUTPUT_DIR}/<파일명>.md` 에 **절대경로**로 저장한다.
4. 완료 후 **저장 경로 + 1~2줄 요약**만 출력한다. 긴 설명 금지.
"""


async def run(transcript_path: str) -> None:
    transcript = Path(transcript_path).expanduser().resolve()
    if not transcript.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {transcript}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=["Read", "Write"],
        permission_mode="acceptEdits",
        # cwd를 input 파일 폴더로 두되, Write는 절대경로로 vault에 씀
        cwd=str(transcript.parent),
    )

    prompt = (
        f"다음 회의 원문을 정리해 주세요.\n"
        f"- 입력 파일 (절대경로): {transcript}\n"
        f"- 저장 디렉토리 (절대경로): {OUTPUT_DIR}\n"
        f"파일명은 `YYYY-MM-DD <주제> 회의록.md` 규칙에 따라 직접 결정하세요."
    )

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python meeting_agent.py <transcript_path>")
        sys.exit(1)
    asyncio.run(run(sys.argv[1]))