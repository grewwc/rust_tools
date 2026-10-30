(function () {
  const engine = __ENGINE__;
  const max = __MAX__;
  const clean = (s) => (s || '').replace(/\s+/g, ' ').trim();
  const unwrap = (href) => {
    try {
      const u = new URL(href, location.href);
      if (/(^|\.)google\./.test(u.hostname) && u.pathname === '/url') {
        const q = u.searchParams.get('q');
        if (q) return q;
      }
      if (/(^|\.)(googleadservices\.com|doubleclick\.net)$/.test(u.hostname) ||
          /\/aclk\?/.test(u.pathname + u.search)) return '';
      return u.href;
    } catch (e) { return ''; }
  };
  const seen = new Set();
  const seenTitles = new Set();
  const results = [];
  const add = (title, href, snippet) => {
    const url = unwrap(href);
    const name = clean(title);
    if (!name || !/^https?:\/\//i.test(url)) return;
    if (seen.has(url) || seenTitles.has(name)) return;
    seen.add(url);
    seenTitles.add(name);
    results.push({ title: name, url: url, snippet: clean(snippet).slice(0, 400) });
  };
  // Baidu mobile renders result titles without an h3 wrapper, and a card can
  // match several container selectors at once. Prefer explicit title nodes,
  // then fall back to the longest anchor inside the card.
  const pickLink = (block) => {
    const selectors = ['h3 a', '.c-title a', '.c-title-text a', 'a.c-title-text', '[class*="title"] a'];
    for (let i = 0; i < selectors.length; i++) {
      const link = block.querySelector(selectors[i]);
      if (link && clean(link.innerText).length >= 4) return link;
    }
    let best = null;
    block.querySelectorAll('a[href]').forEach((link) => {
      const text = clean(link.innerText);
      if (text.length >= 8 && (!best || text.length > clean(best.innerText).length)) best = link;
    });
    return best;
  };
  const pickSnippet = (block, title) => {
    const el = block.querySelector('.c-abstract, .c-span-last, [class*="abstract"], [class*="summary"]');
    let text = clean(el ? el.innerText : block.innerText);
    if (title && text.indexOf(title) === 0) {
      text = text.slice(title.length).trim();
    }
    return text;
  };
  if (engine === 'google') {
    document.querySelectorAll('div.g, div.MjjYud, div[data-hveid]').forEach((block) => {
      const link = block.querySelector('a[href]');
      const heading = block.querySelector('h3');
      const text = block.querySelector('div[data-sncf], div.VwiC3b, span.aCOpRe');
      if (link && heading) add(heading.innerText, link.href, text ? text.innerText : '');
    });
    if (!results.length) {
      document.querySelectorAll('a h3').forEach((heading) => {
        const link = heading.closest('a');
        if (link) add(heading.innerText, link.href, '');
      });
    }
  } else if (engine === 'bing') {
    document.querySelectorAll('li.b_algo').forEach((block) => {
      const link = block.querySelector('h2 a');
      const text = block.querySelector('.b_caption p, p');
      if (link) add(link.innerText, link.href, text ? text.innerText : '');
    });
  } else if (engine === 'baidu') {
    document.querySelectorAll('div[mu], div.c-result, div.result, div.result-op, div.c-container').forEach((block) => {
      const link = pickLink(block);
      if (!link) return;
      const title = clean(link.innerText);
      const mu = (block.getAttribute('mu') || '').trim();
      const href = /^https?:\/\//i.test(mu) ? mu : link.href;
      add(title, href, pickSnippet(block, title));
    });
  } else if (engine === 'duckduckgo') {
    document.querySelectorAll('article[data-testid="result"], div.result').forEach((block) => {
      const link = block.querySelector('a[data-testid="result-title-a"], h2 a, a.result__a');
      const text = block.querySelector('[data-result="snippet"], .result__snippet');
      if (link) add(link.innerText, link.href, text ? text.innerText : '');
    });
  }
  // Last resort, also covers engines whose markup changed: take substantial
  // anchors outside page chrome, grouped by their nearest result card.
  if (!results.length) {
    const nav = 'header, nav, footer, #head, #s_tab, #foot, #rs, .s_tab, .rs, .s-hotsearch-wrapper, .tsn_inner, form';
    const junk = /^(百度首页|百度一下|登录|注册|设置|反馈|更多|下一页|上一页|相关搜索|大家还在搜|首页|搜索)$/;
    document.querySelectorAll('a[href]').forEach((link) => {
      if (link.closest(nav)) return;
      const title = clean(link.innerText);
      if (title.length < 8 || title.length > 250 || junk.test(title)) return;
      const block = link.closest('div[mu], div.c-result, div.result, div.c-container, li, article') || link.parentElement;
      if (!block) return;
      const blockText = clean(block.innerText);
      if (blockText.length < title.length + 12) return;
      add(title, link.href, blockText.replace(title, ''));
    });
  }
  const body = (document.body && document.body.innerText) || '';
  const verification = /captcha|unusual traffic|verify you are human|人机验证|验证码|安全验证|请完成验证|请输入验证码/i.test(body)
    || !!document.querySelector('iframe[src*="recaptcha"], iframe[src*="captcha"], #captcha, .g-recaptcha, [class*="captcha"]');
  const consent = /^consent\./.test(location.hostname) || /before you continue|接受所有|同意并继续/i.test(body);
  const state = results.length ? 'results' : (verification || consent ? 'verify' : 'empty');
  const blocks = document.querySelectorAll('div[mu], div.c-result, div.result, div.result-op, div.c-container').length;
  const anchors = document.querySelectorAll('a[href]').length;
  return JSON.stringify({
    state: state,
    results: results.slice(0, max),
    debug: { blocks: blocks, anchors: anchors, host: location.hostname }
  });
})()
