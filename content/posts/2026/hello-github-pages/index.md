---
title: GitHub Pages로 블로그 열기
description: GitHub Pages로 개인 정적 블로그를 열 때 필요한 저장소 이름, 파일 구조, 배포 설정을 정리합니다.
date: 2026-07-07
category: setup
tags: GitHub Pages, static site
---

## 왜 별도 저장소로 시작했나

회사 계정으로 관리되는 워크스페이스 안에 개인 블로그를 넣으면 원격 저장소와
권한 경계가 흐려집니다. 그래서 홈 디렉터리에 개인 계정의 Pages 저장소를 따로 두었습니다.

## 최소 구조

처음에는 빌드 도구가 없어도 충분합니다. 루트에 `index.html`과
`styles.css`를 두면 GitHub Pages가 그대로 서빙합니다.

```
beengineer500.github.io/
├── index.html
├── styles.css
├── categories/
│   ├── index.html
│   └── setup.html
├── tags/
│   ├── github-pages.html
│   └── static-site.html
├── posts/
│   └── hello-github-pages.html
└── assets/
    └── memo-mark.svg
```

## 다음 결정

글이 많아지면 Markdown 기반 생성기나 RSS를 붙일 수 있습니다. 지금은 운영 비용이 가장 낮은
정적 HTML로 시작하고, 반복 작업이 생기는 시점에 자동화를 추가하는 편이 낫습니다.
