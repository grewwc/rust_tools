const test = require('node:test');
const assert = require('node:assert/strict');
const React = require('react');
const { renderToStaticMarkup } = require('react-dom/server');

test('renders requirement headings, lists, code and GFM tables', async () => {
  const { default: Markdown } = await import('../src/features/ai_node_web_aeolus_card/Markdown.mjs');
  const html = renderToStaticMarkup(React.createElement(Markdown, {
    content: '# 需求共识\n\n- **验收**：`pytest`\n\n```python\nassert True\n```\n\n|目标|结果|\n|---|---|\n|测试|通过|',
  }));
  for (const tag of ['<h1>', '<ul>', '<strong>', '<code>', '<pre>', '<table>']) {
    assert.ok(html.includes(tag), tag);
  }
});

test('does not render raw HTML or executable links', async () => {
  const { default: Markdown } = await import('../src/features/ai_node_web_aeolus_card/Markdown.mjs');
  const html = renderToStaticMarkup(React.createElement(Markdown, {
    content: '<script>alert(1)</script>\n\n[bad](javascript:alert%281%29)\n\n[文档](https://example.com)',
  }));
  assert.ok(!html.includes('<script'));
  assert.ok(!html.includes('href="javascript:'));
  assert.ok(html.includes('href="https://example.com"'));
  assert.ok(html.includes('rel="noopener noreferrer"'));
});
