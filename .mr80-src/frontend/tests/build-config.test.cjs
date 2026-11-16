const test = require('node:test');
const assert = require('node:assert/strict');
const { resolveApiBase } = require('../scripts/lpm.cjs');

test('selects separate development and production API origins', () => {
  assert.equal(resolveApiBase('dev'), 'http://10.37.248.120:8000');
  assert.equal(resolveApiBase('prod'), 'https://aeolus-conciliator.byted.org');
});

test('overrides the selected origin and removes its trailing slash', () => {
  assert.equal(resolveApiBase('dev', 'http://localhost:9000/'), 'http://localhost:9000');
  assert.equal(resolveApiBase('prod', 'https://test.example/'), 'https://test.example');
});

test('rejects invalid origins instead of silently using a default', () => {
  for (const value of ['', 'example.com', 'ftp://example.com', 'https://user:password@example.com',
    'https://example.com/api', 'https://example.com/?token=x', 'https://example.com/#fragment']) {
    assert.throws(() => resolveApiBase('dev', value), /HTTP\(S\) origin/);
  }
  assert.throws(() => resolveApiBase('prod', 'http://example.com'), /HTTPS/);
});

test('rejects unknown profiles including object prototype names', () => {
  for (const profile of ['production', 'staging', 'toString', '__proto__', undefined]) {
    assert.throws(() => resolveApiBase(profile), /profile must be dev or prod/);
  }
});
