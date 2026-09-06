// mermaid 도식 렌더링. 도식이 있는 글에만 주입된다 (scripts/build.mjs의 mermaidExtras).
// 테마가 바뀌면 다시 그린다 - mermaid는 초기화 시점의 테마를 한 번만 읽기 때문이다.
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";

const root = document.documentElement;
const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");
const blocks = Array.from(document.querySelectorAll("pre.mermaid"));

// mermaid가 렌더링하며 원본을 지우므로, 다시 그릴 수 있게 보관해둔다.
for (const block of blocks) {
  block.dataset.src = block.textContent;
}

function isDarkTheme() {
  const explicit = root.getAttribute("data-theme");
  if (explicit) return explicit === "dark";
  return darkQuery.matches;
}

function reveal() {
  root.classList.remove("mermaid-pending");
}

async function render() {
  if (!blocks.length) return reveal();

  for (const block of blocks) {
    block.removeAttribute("data-processed");
    block.textContent = block.dataset.src;
  }

  mermaid.initialize({
    startOnLoad: false,
    securityLevel: "strict",
    theme: isDarkTheme() ? "dark" : "default",
  });

  try {
    await mermaid.run({ nodes: blocks });
  } catch (error) {
    // 실패하면 원본 텍스트라도 보이게 둔다.
    console.error("mermaid 렌더링 실패", error);
  } finally {
    reveal();
  }
}

render();

new MutationObserver(render).observe(root, { attributeFilter: ["data-theme"] });

// 사용자가 테마를 직접 고르지 않았을 때만 OS 설정 변화를 따라간다.
darkQuery.addEventListener("change", () => {
  if (!root.getAttribute("data-theme")) render();
});
