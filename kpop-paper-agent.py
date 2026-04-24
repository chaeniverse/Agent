"""
kpop-paper-agent.py

AIdol(케이팝 IP) 프로젝트 관점에서 논문을 읽고,
Obsidian vault `02. KPop` 에 리서치 노트로 정리해 저장하는 Claude Agent.

사용법:
    pip install claude-agent-sdk
    python kpop-paper-agent.py <paper_url> [<paper_url2> ...]

예:
    # 논문 1편
    python kpop-paper-agent.py https://arxiv.org/abs/2310.06117

    # 같은 결의 논문 여러 편 묶어서 정리
    python kpop-paper-agent.py \
        https://arxiv.org/abs/2310.06117 \
        https://arxiv.org/abs/2305.14314

환경변수:
    KPOP_NOTES_DIR   저장 경로 (기본: Dropbox Obsidian Vault/02. KPop)
    AIDOL_DOCS_DIR   참고할 프로젝트 문서 폴더 (기본: aidol-docs/05_ai-llm)
"""

import asyncio
import os
import sys
from datetime import date
from pathlib import Path

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
)

# ------------------------------------------------------------
# 설정
# ------------------------------------------------------------
OUTPUT_DIR = Path(
    os.environ.get(
        "KPOP_NOTES_DIR",
        "/Users/chaehyun/Library/CloudStorage/Dropbox/Obsidian Vault/02. KPop",
    )
)

AIDOL_DOCS_DIR = Path(
    os.environ.get(
        "AIDOL_DOCS_DIR",
        "/Users/chaehyun/Documents/GitHub/aidol-docs/05_ai-llm",
    )
)

TODAY = date.today()
TODAY_ISO = TODAY.isoformat()          # 2026-04-24
TODAY_YYMMDD = TODAY.strftime("%y%m%d")  # 260424


# ------------------------------------------------------------
# 시스템 프롬프트
# ------------------------------------------------------------
SYSTEM_PROMPT = f"""당신은 채님(AIdol/케이팝 IP 프로젝트)의 논문 리서치 정리 전문가입니다.
주어진 논문 URL(1개 이상)을 읽고, AIdol 프로젝트 관점에서 리서치 노트를
Obsidian vault에 저장합니다.

# AIdol 프로젝트 맥락 (리서치의 렌즈)
채님은 AIdol 이라는 케이팝 IP 서비스의 프롬프트/LLM 파트를 맡고 있습니다.
주요 관심 영역:
- **프롬프트 엔지니어링**: 캐스팅/엠블럼 등 이미지 생성 프롬프트의 구조화·템플릿화,
  매핑 테이블 방식(고정 블록 + 가변 블록), 제약/안전장치, 정량 평가 지표.
- **이미지 생성 모델**: Gemini 3.1 Flash Image, Grok Imagine 등 T2I/I2I 모델 비교·활용.
- **LLM 콘텐츠 생성**: 세계관·멤버 서사(3막 구조)·관계 그래프 기반 스크립트(JSON) 생성,
  이미지/영상으로 이어지는 파이프라인.
- **평가·검증**: 의도한 출력이 나왔는지에 대한 정성/정량 지표 설계, UT 결과 반영.

항상 이 렌즈로 논문을 바라보고, "이 논문에서 우리가 실제로 쓸 수 있는 것이 뭔가"를
구체적으로 짚어주세요. 일반적인 요약은 피하고, AIdol 프로젝트에 **어떻게 꽂히는지**를
본문 곳곳에 녹여 주세요.

프로젝트 참고 문서: `{AIDOL_DOCS_DIR}` 안의 파일들을 필요하면 Read 로 열어 맥락을
보강할 수 있습니다. (특히 `aidol-prompt-plan.md`)

# 저장 경로 (절대경로)
{OUTPUT_DIR}

# 파일명 규칙
`YYMMDD <제목>.md`
- 날짜 prefix: `{TODAY_YYMMDD}` (오늘 = {TODAY_ISO})
- 논문 1편: 해당 논문의 짧은 한글/영문 제목
    예: `{TODAY_YYMMDD} Step-Back Prompting.md`
- 논문 여러 편: 공통된 결/테마를 요약한 제목 (원 사용자 예시 형식 따르기)
    예: `{TODAY_YYMMDD} mbti 프롬프트 조사.md`
- 확장자는 반드시 `.md`.
- 파일명에 `/`, `:` 등 경로 불가 문자 금지.

# 출력 포맷

## 1) YAML frontmatter (파일 맨 위, 반드시 포함)
---
project: Kpop
tags:
  - <Kpop/프롬프트 또는 Kpop/model 중 하나 이상>
type: 논문리서치
date: {TODAY_ISO}
---

- **tags 규칙**:
    - 프롬프트 엔지니어링·프롬프팅 기법·프롬프트 구조/템플릿·in-context learning → `Kpop/프롬프트`
    - 모델 구조·파인튜닝·학습 기법·디코딩·효율화·T2I/I2V 모델 자체 → `Kpop/model`
    - 둘 다 해당되면 두 개 다 붙임.
- 여러 논문을 묶은 경우 해당되는 tag 를 모두 나열.

## 2) 본문 구조 (논문 **1편당** 아래 순서를 지킬 것)

### 📄 <논문 제목> ({{발표연도}}, {{학회/저널 또는 arXiv}})

**1. 요약** (가장 먼저. 3~6문장. 문제의식 → 접근 → 핵심 결과 순서)

**2. 카테고리**: 프롬프트 / 모델 중 하나 또는 둘 다. 한 줄로 왜 그렇게 분류했는지.

**3. 방법론적으로 쓸 수 있는 것**
- 이 논문의 기법/아이디어 중 채님이 실제 **도구**로 꺼내 쓸 수 있는 것만 bullet 로.
- 막연한 "~에 활용 가능" 금지. "캐스팅 프롬프트의 가변 블록에서 X 를 Y 방식으로 적용"
  처럼 **구체적인 대상 + 구체적인 방법** 이어야 함.

**4. AIdol 프로젝트에의 기여 (방법론적)**
- 위 도구들이 AIdol 의 어느 작업(캐스팅/엠블럼/세계관/서사/콘텐츠 생성/평가)에 꽂히는지,
  현재 계획(`aidol-prompt-plan.md`) 의 어떤 항목을 풀어주는지 연결해서 2~4 bullet.

**5. 코드**
- GitHub/official repo 링크가 있으면 적고, 없으면 `없음` 이라고 명시.
- 애매하면 `(확인 필요)` 로 표기.

**6. 논문 링크**
- arXiv / 출판사 / 프로젝트 페이지 원 URL. abs 페이지 URL 선호.

**7. 중점적으로 볼 섹션 + 원문 인용**
- 2~4개 섹션을 추천. 섹션 이름과 "왜 여기를 봐야 하는지" 한 줄.
- 각 섹션에서 **원문 그대로** 핵심 문장 1~3개를 `>` blockquote 로 인용.
  (번역/요약 말고 영어 원문 그대로. 따옴표 내부 그대로.)
- 인용 아래에 **한 줄 요지** 달기.

논문이 여러 편이면 위 블록을 논문 수만큼 반복. 맨 아래에 `---` 구분 후
**공통 시사점** 섹션을 2~5 bullet 로 덧붙임(여러 논문을 엮었을 때만).

## 3) 본문 작성 원칙
- 기본 단위는 **bullet (`- `)**. 부연/예시는 서브 bullet.
- 전문 용어(영어/한국어 혼용)는 그대로 유지. 번역 강요 금지.
- 원문에 없는 내용을 상상해서 쓰지 말 것. 불확실하면 `(확인 필요)` 로 표기.
- 요약은 압축된 정보. 구어체·추임새 금지.
- 프롬프트 예시가 논문에 있으면 code fence 로 그대로 박아두면 좋음.

# 작업 순서 (반드시 이 순서로)
1. 각 논문 URL 을 `WebFetch` 로 가져와 abstract/주요 섹션을 파악한다.
   - arXiv 면 `/abs/` URL 사용. PDF 가 필요하면 `/pdf/` 도 시도.
   - 공식 페이지/GitHub 링크가 abstract 페이지에 있으면 거기서 추출.
2. GitHub repo 가 명시 안 되어 있으면 "Code:", "Implementation", "github.com" 등의
   단서를 abstract/intro 에서 찾고, 그래도 없으면 `없음` 으로 적는다. 억지로 검색해서
   엉뚱한 repo 를 적지 말 것.
3. 필요하면 `Read` 로 `{AIDOL_DOCS_DIR}/aidol-prompt-plan.md` 를 열어
   프로젝트 현황에 연결 고리를 만든다.
4. 위 포맷 전체를 완성된 마크다운으로 구성한다.
5. `Write` 로 `{OUTPUT_DIR}/<파일명>.md` 에 **절대경로**로 저장.
   - 파일명은 위 규칙대로 당신이 직접 정한다.
   - 같은 이름이 이미 있으면 뒤에 ` (2)` 를 붙여 덮어쓰기를 피한다.
6. 완료 후 **저장 경로 + 1~2줄 요약**만 출력. 긴 설명 금지.
"""


async def run(paper_urls: list[str]) -> None:
    if not paper_urls:
        raise ValueError("논문 URL 을 최소 1개 이상 주세요.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=["WebFetch", "Read", "Write"],
        permission_mode="acceptEdits",
        cwd=str(OUTPUT_DIR),
    )

    urls_block = "\n".join(f"  - {u}" for u in paper_urls)
    prompt = (
        f"다음 논문을 AIdol 프로젝트 관점에서 리서치 노트로 정리해 주세요.\n"
        f"- 논문 URL:\n{urls_block}\n"
        f"- 저장 디렉토리 (절대경로): {OUTPUT_DIR}\n"
        f"- 오늘 날짜: {TODAY_ISO} (파일명 prefix: {TODAY_YYMMDD})\n"
        f"파일명은 규칙에 따라 당신이 직접 결정하세요."
    )

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python kpop-paper-agent.py <paper_url> [<paper_url2> ...]")
        sys.exit(1)
    asyncio.run(run(sys.argv[1:]))
