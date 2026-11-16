const { spawnSync } = require('node:child_process');
const { cpSync, existsSync, writeFileSync } = require('node:fs');
const path = require('node:path');
const profiles = require('../build.config.json');

const frontendRoot = path.resolve(__dirname, '..');

function resolveApiBase(profile, override) {
  if (!Object.hasOwn(profiles, profile)) {
    throw new Error('Build profile must be dev or prod.');
  }
  let url;
  try {
    url = new URL(override ?? profiles[profile].apiBase);
  } catch {
    throw new Error('API base must be a valid HTTP(S) origin.');
  }
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password
    || url.pathname !== '/' || url.search || url.hash) {
    throw new Error('API base must be an HTTP(S) origin without credentials, path, query or fragment.');
  }
  if (profile === 'prod' && url.protocol !== 'https:') {
    throw new Error('Production API base must use HTTPS.');
  }
  return url.origin;
}

if (require.main === module) {
  try {
    const [profile, action, ...extra] = process.argv.slice(2);
    if (extra.length || !['start', 'build', 'pack'].includes(action)) {
      throw new Error('Usage: node scripts/lpm.cjs <dev|prod> <start|build|pack>');
    }
    const apiBase = resolveApiBase(profile, process.env.MEEGLE_CARD_API_BASE);
    const pluginRoot = path.resolve(process.env.MEEGLE_PLUGIN_ROOT || frontendRoot);
    if (!existsSync(path.join(pluginRoot, 'plugin.config.json'))) {
      throw new Error('Plugin root must contain an existing plugin.config.json.');
    }
    if (pluginRoot !== frontendRoot) {
      for (const file of ['src', 'scripts', 'tests', 'build.config.json', 'package.json',
        'package-lock.json', 'tsconfig.json', 'global.d.ts']) {
        cpSync(path.join(frontendRoot, file), path.join(pluginRoot, file), { recursive: true });
      }
    }
    writeFileSync(path.join(pluginRoot, 'src/build-config.json'),
      `${JSON.stringify({ apiBase }, null, 2)}\n`);
    console.log(`[card] ${profile} API_BASE=${apiBase}`);
    const args = ['--cwd', pluginRoot, action === 'pack' ? 'build' : action];
    if (action !== 'start') args.push('--source-type', 'local');
    if (action === 'pack') args.push('--zip');
    const result = spawnSync('lpm', args, { cwd: pluginRoot, stdio: 'inherit' });
    if (result.error) {
      throw new Error(`Cannot run lpm (${result.error.code}); install @lark-project/cli and add it to PATH.`);
    }
    process.exitCode = result.status ?? 1;
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}

module.exports = { resolveApiBase };
