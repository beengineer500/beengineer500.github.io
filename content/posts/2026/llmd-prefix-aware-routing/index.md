---
title: 균등 분산과 캐시 지역성은 다른 목표다 - llm-d prefix-aware 라우팅 실측
description: Kubernetes Service 직결과 load-only 라우터와 prefix-aware 라우터에 같은 프리픽스 반복 워크로드를 흘려 backend request 분포와 prefix-cache hit rate를 비교한 기록
date: 2026-09-06
category: llm
tags: llm-d, prefix-caching, llm-serving, kubernetes
---

<nav id="manual-toc" class="manual-toc" aria-label="목차">

### 목차

**[1부. 캐시를 모르는 분산](#section-1)**

- [1.1 균등 분산은 cache가 있는 backend를 모른다](#s1-1)
- [1.2 세 요청 경로](#s1-2)

**[2부. 세 경로의 측정 결과](#section-2)**

- [2.1 벤치 결과 - duration이 아니라 도착률 미달을 봐야 한다](#s2-1)
- [2.2 균등한 48:52 분포, 그런데 hit rate는 33%·30%·65%](#s2-2)

**[3부. 라우터가 실제로 한 일](#section-3)**

- [3.1 프리픽스가 KV 이벤트를 거쳐 backend 선택으로 이어지는 경로](#s3-1)
- [3.2 세 가지 증거를 함께 봐야 하는 이유](#s3-2)
- [3.3 median TTFT 539.42ms, P99 TTFT 4,816.05ms](#s3-3)

**[전체 흐름 정리](#section-4)**

**[LLM serving 인사이트](#section-5)**

**[출처](#section-6)**

</nav>

---

## 1부. 캐시를 모르는 분산

<a id="s1-1"></a>

### 1.1 균등 분산은 cache가 있는 backend를 모른다

Kubernetes Service의 기본 로드밸런싱은 백엔드 파드의 상태를 보지 않습니다. 

큐 길이도, GPU 메모리 점유율도, 그 파드가 이미 어떤 프리픽스를 KV 캐시에 들고 있는지도 모릅니다. 요청이 왔을 때 다음 파드를 고를 뿐입니다. 그래서 같은 프리픽스를 반복해서 보내는 워크로드(멀티턴 대화, few-shot 프롬프트, RAG의 고정 컨텍스트 등)에서는 매번 다른 파드로 요청이 튈 수 있고, 그때마다 그 파드는 프리픽스를 처음부터 다시 계산합니다.

이 문제를 풀려는 것이 prefix-aware 라우팅입니다. 요청의 프리픽스를 토큰 단위로 해싱해 어느 백엔드가 그 블록을 이미 캐시에 들고 있는지 추적하고, 그 파드로 우선 보냅니다. 이번 글은 같은 Qwen3-14B BF16 배포(replica 2개)에 같은 프리픽스 반복 워크로드를 세 가지 경로로 흘려 무엇이 달라지는지 봅니다.

<a id="s1-2"></a>

### 1.2 세 요청 경로

같은 두 BF16 replica 앞에 세 가지 진입점을 뒀습니다. 라우터의 `modelServers.matchLabels`가 `app: vllm-qwen3-14b`로 이 두 replica를 잡았고, svc의 타깃도 같은 BF16 Service였습니다. 세 경로 모두 같은 백엔드로 라우팅하도록 구성했습니다.

```
① Service 직결
vllm bench serve
      │
      ▼
ClusterIP Service (vllm-qwen3-14b:8000)
      │
 ┌────┴────┐
 ▼         ▼
replica    replica
7w92m      pm7wp


② load-only 라우터
vllm bench serve
      │
      ▼
Envoy sidecar
      │
      ▼
EPP (queue-scorer + kv-cache-utilization-scorer)
      │
 ┌────┴────┐
 ▼         ▼
replica    replica
7w92m      pm7wp


③ prefix-aware 라우터
vllm bench serve
      │
      ▼
Envoy sidecar
      │
      ▼
tokenizer 사이드카 (vllm-openai-cpu)
      │
      ▼
EPP (prefix-cache-scorer w3 + queue-scorer w1 + kv-cache-utilization-scorer w1)
      │
 ┌────┴────┐
 ▼         ▼
replica    replica
7w92m      pm7wp
   │          │
   └── KV event :5557 ──┴── KV event :5557 ──▶ EPP

세 경로 모두 같은 두 replica(7w92m, pm7wp)로 수렴한다
```

- ①은 파드 상태를 전혀 참조하지 않는 Kubernetes Service 라운드로빈입니다. 
- ②는 Envoy와 EPP를 거치지만 큐 길이와 KV 캐시 이용률만 보고 프리픽스는 모릅니다. 
- ③은 여기에 tokenizer 사이드카와 각 replica가 `:5557`로 발행하는 KV 이벤트 구독이 더해져, EPP가 "이 프리픽스는 어느 파드에 있는가"를 알고 스코어링에 반영합니다(`prefix-cache-scorer` weight 3으로 다른 두 스코어러 weight 1+1보다 가중치가 높습니다).

---

## 2부. 세 경로의 측정 결과

<a id="s2-1"></a>

### 2.1 벤치 결과 - duration이 아니라 도착률 미달을 봐야 한다

세 워크로드의 벤치 결과부터 보겠습니다. 조건은 동일합니다: 
- `prefix_repetition` 데이터셋
- `prefix_len=3072`
- `num_prefixes=160`
- `num_prompts=480`(프리픽스당 3회 재사용)
- `request_rate=10.0`
- `max_concurrency=32`
- `temperature=0`


| 지표 | svc 직결 | load-only | prefix-aware |
|---|---:|---:|---:|
| duration (s) | 84.014 | 87.889 | 51.903 |
| request throughput (req/s) | 5.713 | 5.461 | 9.248 |
| output throughput (tok/s) | 182.83 | 174.77 | 295.94 |
| total token throughput (tok/s) | 18,099.88 | 17,301.79 | 29,297.64 |
| max_concurrent_requests | 44 | 48 | 64 |
| max_output_tokens_per_s | 417 | 554 | 927 |
| mean TTFT (ms) | 731.81 | 1,341.93 | 943.22 |
| median TTFT (ms) | 622.95 | 1,172.08 | 539.42 |
| P99 TTFT (ms) | 3,408.49 | 4,820.64 | 4,816.05 |
| median TPOT (ms) | 156.28 | 135.90 | 54.80 |
| P99 TPOT (ms) | 210.71 | 222.77 | 208.62 |
| median E2EL (ms) | 5,434.31 | 5,702.79 | 2,441.16 |
| P99 E2EL (ms) | 9,715.07 | 9,589.20 | 8,475.36 |


prefix-aware가 throughput과 median 지표에서 가장 좋아 보입니다. 그런데 이 표만 보고 "prefix-aware 라우팅이 효과가 있었다"고 결론 내리면 성급합니다. latency와 throughput은 결과일 뿐, 그 결과가 "캐시를 잘 맞혀서"인지 "다른 요인 때문"인지는 이 표 자체로는 구분되지 않습니다.

세 결과 모두 목표 `request_rate=10.0`, `max_concurrency=32`로 실행했지만, 실제 달성 request throughput은 목표 대비 svc 직결 57.1%(5.713 / 10.0), load-only 54.6%(5.461 / 10.0), prefix-aware 92.5%(9.248 / 10.0)에 그쳤습니다. `max_concurrent_requests`도 44 / 48 / 64로 세 결과가 서로 다릅니다.

**해석**: `max_concurrency=32` 상한이 걸려 있으면 요청 도착이 처리 속도에 종속되는 closed-loop 성격을 띱니다. 이 조건에서 throughput 차이는 "부하를 얼마나 밀어 넣었는가"가 아니라 "얼마나 빨리 비웠는가"를 반영한다고 볼 수 있습니다. 다만 워크로드마다 seed와 캐시 상태가 다르므로 이것이 좋은 비교는 아닙니다. 그래서 backend 쪽 관측이 추가로 필요합니다.

<a id="s2-2"></a>

### 2.2 균등한 48:52 분포, 그런데 hit rate는 33%·30%·65%

세 워크로드 각각에서 backend pod별 request 수와 vLLM의 `prefix_cache_queries_total` / `prefix_cache_hits_total` 카운터 delta(전/후 스냅샷 차분)를 봅니다. 

aggregate hit rate는 pod별 비율을 평균한 값이 아니라 **`합계 hits / 합계 queries`**로 계산했습니다.


| 경로 | pod 7w92m req (비중) | pod pm7wp req (비중) | 합계 req | queries 합계 | hits 합계 | aggregate hit rate |
|---|---:|---:|---:|---:|---:|---:|
| svc 직결 | 235 (49.0%) | 245 (51.0%) | 480 | 1,505,284 | 491,488 | **32.65%** |
| load-only | 245 (51.0%) | 235 (49.0%) | 480 | 1,505,296 | 451,392 | **29.99%** |
| prefix-aware | 231 (48.1%) | 249 (51.9%) | 480 | 1,505,293 | 973,728 | **64.69%** |

세 경로 모두 request가 두 파드에 48~52% 범위로 고르게 나뉩니다. request 분포만 보면 세 경로가 거의 똑같습니다. 그런데 aggregate prefix-cache hit rate는 svc 직결 32.65%, load-only 29.99%, prefix-aware 64.69%로 두 배 이상 차이 납니다.

**해석**: 같은 수의 요청이 같은 비율로 두 파드에 나뉘어도, 어느 프리픽스가 어느 파드로 갔는지에 따라 캐시 재사용률이 크게 달라질 수 있다는 뜻으로 읽습니다.

**아직 모르는 것**: 세 경로의 queries 합계가 1,505,284 / 1,505,296 / 1,505,293으로 4~16 토큰만큼씩 미세하게 다릅니다. vLLM의 `prefix_cache_queries_total`은 이 스냅샷에서 `prompt_tokens_total`과 정확히 같은 값으로, request 단위가 아니라 **토큰(블록) 단위 카운터**입니다. 이 정도 편차가 스냅샷 경계에서 다른 트래픽이 섞여 들어간 흔적인지는 이 데이터만으로 확정할 수 없습니다.

---

## 3부. 라우터가 실제로 한 일

<a id="s3-1"></a>

### 3.1 프리픽스가 KV 이벤트를 거쳐 backend 선택으로 이어지는 경로

prefix-aware 경로에서 어떤 컴포넌트가 무엇을 하는지 좀 더 뜯어보겠습니다.

```
요청 프리픽스 (3072 토큰)
      │
      ▼
tokenizer 사이드카 (vllm-openai-cpu)
      │
      ▼
token-producer 플러그인
      │
      ▼
precise-prefix-cache-producer (blockSizeTokens=16, hashSeed=42)
      │   ◀┄┄ 관찰: 구독 ┄┄ replica 7w92m (KV event :5557)
      │   ◀┄┄ 관찰: 구독 ┄┄ replica pm7wp (KV event :5557)
      │
      ┆┄┄▶ 가설: 블록 단위 인덱스 (admission/eviction)
      ▼
prefix-cache-scorer (weight 3)
      │
      ▼
max-score-picker
      │
      ▼
backend 선택
      │
      ▼
vLLM replica의 prefix cache hit/miss
```

요청이 들어오면 tokenizer 사이드카가 프리픽스를 토큰화하고, `precise-prefix-cache-producer`가 16토큰 블록 단위로 해싱(`hashSeed: "42"`)합니다. 

두 replica가 `:5557`로 발행하는 KV 이벤트를 구독해 어느 블록이 어느 파드에 있는지 인덱스로 유지하고, 이 인덱스를 `prefix-cache-scorer`(weight 3, 다른 두 스코어러 weight 1+1보다 지배적)가 점수화해 `max-score-picker`가 최종 backend를 고릅니다. 점선은 이번 데이터로 직접 확인하지 못한 인덱스 내부 동작(가설)입니다.

**EPP 카운터**: 아래 값은 480요청 벤치마크가 아니라, 그보다 앞서 실행된 **별도의 16요청 동작 확인 프로브**(22:40:54 → 22:42:37 UTC, 벤치마크는 22:46 이후 시작) 창에서 찍힌 스냅샷입니다. 프로브는 `uuid4().hex + " " + ("cache routing example " * 1024)` 형태로 프리픽스를 공유하는 completion 16개를 `max_tokens=4`로 보낸 것입니다. 창이 다르므로 이 지표를 480요청 벤치 결과와 같은 표에 넣어 인과를 주장하지 않습니다.

| metric | before | after (delta) |
|---|---:|---:|
| `kv_cache_events_messages_received_total` (7w92m) | (없음) | 17 |
| `kv_cache_index_admissions_total` | 0 | 194 |
| `kv_cache_index_evictions_total` | 0 | 207 |
| `kv_cache_index_lookup_requests_total` | 0 | 16 |
| `kv_cache_index_lookup_hits_total` | 0 | 2,910 |
| `kv_cache_events_dedup_removed_hashes_forwarded_total` | 0 | 207 |
| `epp_scheduler_attempts_total{status="success"}` (7w92m) | (없음) | 16 |
| `epp_request_total` | (없음) | 16 |

**관찰**: 같은 프로브 창의 backend 카운터 delta도 확인했습니다. **7w92m만 16 request 전부**를 받았고(prompt/queries 49,702, hits 46,560 → 93.68%), pm7wp는 0건입니다. 16개 요청이 한 파드에 100% 몰린 것입니다.

**관찰**: 반면 앞서 본 prefix-aware 경로 벤치(480요청, 프리픽스 160종)에서는 request가 231:249, 즉 48.1%:51.9%로 어느 한쪽에 쏠리지 않고 고르게 나뉘었습니다.

**해석**: 프로브는 16개 요청이 사실상 같은 프리픽스 하나를 공유해 최적 목적지가 파드 하나로 고정됩니다. 반면 벤치는 서로 다른 160개 프리픽스를 썼기 때문에, 각 프리픽스가 자신의 최적 목적지(먼저 그 프리픽스를 캐시에 들고 있는 파드)를 따라 두 파드에 나뉘어 자리 잡으면 request 총량 자체는 균등해질 수 있습니다. 이 대비가 "균등 분포는 cache affinity 부재를 뜻하지 않는다"는 이 글의 논지를 가장 직접적으로 뒷받침합니다.

**해석 - 블록 단위로만 읽는다**: `lookup_hits / lookup_requests = 2,910 / 16`을 request-level hit rate로 계산하지 않습니다. `lookup_requests`는 조회 호출 횟수(요청당 1회)이고, `lookup_hits`는 **16토큰 블록 단위** 매칭 수입니다. 프로브 요청 하나가 평균 3,106 토큰(49,702 / 16)이므로 요청당 약 194개 블록이 나오고, 그중 평균 약 182개 블록이 인덱스에서 매칭됐다고 블록 단위로만 읽습니다.

**해석 - admission과 첫 요청 인덱싱 가설**: `admissions 194`가 요청당 블록 수(약 194)와 거의 같다는 점은 "첫 요청분 블록이 인덱스에 등록되고 이후 요청은 대부분 조회로 처리됐다"는 **가설**이지, 이 데이터가 직접 증명하는 사실은 아닙니다.

**아직 모르는 것**: `evictions 207`이 `admissions 194`보다 많습니다. 이는 인덱스 churn이 있었다는 신호이지만, 원인(용량 압박인지 TTL인지 다른 조건인지)은 이 데이터로 알 수 없습니다.

<a id="s3-2"></a>

### 3.2 세 가지 증거를 함께 봐야 하는 이유

```
                  A. backend request 분포
                     48~52%로 균등
                 ╱                     ╲
   A─B만 보면:                          A─C만 보면:
   "세 경로가 똑같다"는 오독              "EPP가 도는지"만 확인,
                                        성능 효과는 모름
               ╱                           ╲
B. prefix-cache hit rate ── B─C만 보면: ── C. router event·index metric
   32.65% / 29.99% / 64.69%   "파드 배치는     messages 17, admissions 194,
                               그냥 우연"       lookup req 16 / hits 2,910(블록)
                               이라는 오독
```

**캡션**: 한 꼭짓점만 보면 각각 다른 방향으로 오독합니다. request 분포만 보면 세 경로가 거의 동일해 보이지만 실제로는 어느 프리픽스가 어디로 갔는지가 다릅니다. hit rate만 보면 왜 그런지(EPP가 실제로 인덱스를 유지하고 스코어링에 반영했는지)를 설명하지 못합니다. EPP 이벤트·인덱스 카운터만 보면 "메커니즘이 동작 중"이라는 integration 증거는 되지만 480요청 벤치의 성능 수치와는 다른 창이라 그 자체로 성능 개선을 증명하지 않습니다. 세 개를 겹쳐 봐야 "prefix-aware 경로가 backend cache affinity를 실제로 바꿨다"는 그림이 완성됩니다.

<a id="s3-3"></a>

### 3.3 median TTFT 539.42ms, P99 TTFT 4,816.05ms

**관찰**: prefix-aware의 request throughput은 9.248 req/s, total token throughput은 29,297.64 tok/s로 세 경로 중 가장 높습니다. median TTFT도 539.42ms로 세 경로 중 가장 낮게 관측됐습니다. 반면 P99 TTFT는 4,816.05ms로, load-only의 4,820.64ms와 사실상 같고 svc 직결의 3,408.49ms보다 오히려 높습니다.

**해석**: median과 P99가 갈라지는 것 자체는 이상하지 않습니다. median은 전형적인 요청을 대표하고, P99는 꼬리에 있는 소수의 느린 요청을 대표합니다. 캐시를 맞힌 요청의 TTFT가 크게 줄어 median을 끌어내리고, 캐시를 못 맞히거나(cold 프리픽스, 최초 진입 블록) Envoy·EPP·tokenizer 홉의 오버헤드가 붙는 소수의 요청이 P99를 결정한다고 설명할 수 있지만, 요청별로 캐시 hit/miss와 TTFT를 짝지은 기록이 없어 확인되지 않았습니다.

**아직 모르는 것**: 어느 요청이 캐시를 맞혔고 그 요청들의 TTFT가 실제로 낮았는지, 반대로 P99를 만든 소수의 요청이 캐시 미스였는지는 이 데이터로 검증할 수 없습니다.

---

## 전체 흐름 정리

```
같은 프리픽스 반복 워크로드 (prefix_len=3072, num_prefixes=160, num_prompts=480)
      │
      ▼
세 경로로 흘려보냄
   ① Service 직결        ② load-only 라우터        ③ prefix-aware 라우터
      │                      │                          │
      ▼                      ▼                          ▼
backend request 분포 - 세 경로 모두 48~52%로 균등 (겉보기엔 동일)
      │
      ▼
그런데 aggregate prefix-cache hit rate는 32.65% / 29.99% / 64.69%로 갈림
      │
      ▼
EPP 이벤트·인덱스 카운터로 확인 - KV event 구독(messages 17), 인덱스 admission(194)·lookup(16/2,910)이 실제로 동작
      │
      ▼
결과 - prefix-aware가 request throughput 9.248 req/s로 가장 높고 median TTFT도 539.42ms로 가장 낮음
      단 P99 TTFT(4,816.05ms)는 svc 직결(3,408.49ms)보다 오히려 높음 - median과 tail은 다른 이야기
```

| 숫자 | 뜻 |
|---|---|
| 48~52% | 세 경로 모두 두 replica에 고르게 간 request 비중 |
| 32.65% / 29.99% / 64.69% | svc 직결 / load-only / prefix-aware의 aggregate prefix-cache hit rate |
| 5.713 / 5.461 / 9.248 req/s | 세 경로의 실제 달성 request throughput (목표는 10.0) |
| 539.42ms | prefix-aware의 median TTFT, 세 경로 중 가장 낮음 |
| 4,816.05ms | prefix-aware의 P99 TTFT, svc 직결 3,408.49ms보다 높음 |
| 16토큰 | EPP 프리픽스 인덱스의 블록 크기 (blockSizeTokens) |

## LLM serving 인사이트

- **load balancing과 cache-aware routing은 목적 함수가 다릅니다.**
  - load-only는 큐 길이와 KV 이용률을 평평하게 만드는 것이 목표고, prefix-aware는 같은 프리픽스를 같은 파드로 모으는 것이 목표입니다. 두 목표는 종종 충돌합니다. 
  - 프리픽스 지역성을 우선하면 특정 파드에 부하가 쏠릴 수 있습니다. 이번 실험에서도 프리픽스가 1종이었던 16요청 프로브에서는 7w92m에 100%가 쏠렸지만, 프리픽스가 160종이었던 480요청 prefix-aware 벤치에서는 231:249(48.1%:51.9%)로 쏠리지 않았습니다. 
  - 이 두 관찰을 이어 보면, 프리픽스 지역성과 부하 균등이 충돌하는 정도는 워크로드의 프리픽스 다양성에 달려 있다는 **해석**이 가능합니다.
- **균등한 request 분포가 cache affinity 부재를 뜻하지 않습니다** 
  - 이번 실험에서는 세 경로 모두 request 수가 48~52%로 균등했지만 aggregate hit rate는 32.65% / 29.99% / 64.69%로 갈렸습니다. 
  - 총량이 같아도 어떤 프리픽스가 어디로 갔는가가 다르면 캐시 재사용률은 전혀 다른 이야기가 됩니다.
- **backend request count, prefix-cache hit rate, router event/index metric을 함께 봐야 합니다.** 
  - 이 셋 중 하나만 보면 서로 다른 방향으로 결과 해석에 오류가 발생할 수 있습니다.
- **cold/warm cache, seed, 실행 순서가 benchmark 해석을 바꿉니다.** 같은 조건이라도 어떤 캐시 상태에서 출발했는지에 따라 같은 라우팅 로직의 관측값이 달라질 수 있습니다.

---

## 출처

- llm-d inference scheduler v0.10.0 (commit `71f4f0999f95b96c49a9d0c4afbd18dfdb943c26`) - https://github.com/llm-d/llm-d-inference-scheduler
- EPP 이미지 `ghcr.io/llm-d/llm-d-router-endpoint-picker:v0.10.0`
- 벤치마크 도구 `vllm bench serve` (vLLM v0.11.0) - https://github.com/vllm-project/vllm
- tokenizer 사이드카 이미지 `vllm/vllm-openai-cpu:v0.19.1`
- 모델 `Qwen/Qwen3-14B` - https://huggingface.co/Qwen/Qwen3-14B
- 측정값은 직접 구성한 Kubernetes 실습 환경에서 `vllm bench serve`와 각 파드의 `/metrics` 전후 스냅샷 차분으로 얻었습니다.
