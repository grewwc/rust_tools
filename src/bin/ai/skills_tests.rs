use super::*;
use rust_tools::cw::SkipSet;
use std::io::Write;

fn write_zip_package(path: &Path, entries: &[(&str, &str)]) {
    let file = fs::File::create(path).unwrap();
    let mut zip = zip::ZipWriter::new(file);
    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for (name, content) in entries {
        zip.start_file(name, options).unwrap();
        zip.write_all(content.as_bytes()).unwrap();
    }
    zip.finish().unwrap();
}

#[test]
fn seed_skills_dir_creates_dir_but_does_not_copy_builtins() {
    let dir = std::env::temp_dir().join(format!("rust-tools-skills-{}", uuid::Uuid::new_v4()));
    ensure_seeded_skills_dir(&dir).unwrap();
    let skills = load_skills_from_dir(&dir);
    assert_eq!(skills.len(), 0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_all_skills_loads_user_skill_without_builtins() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let home = std::env::temp_dir()
        .join(format!("rust-tools-home-{}", uuid::Uuid::new_v4()))
        .display()
        .to_string();
    let old_home = std::env::var("HOME").ok();
    unsafe {
        std::env::set_var("HOME", &home);
    }

    let dir = skills_dir();
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("custom.skill"),
        r#"---
name: custom-skill
description: custom
priority: 1
---

custom"#,
    )
    .unwrap();

    let skills = load_all_skills();
    let mut names = SkipSet::new(8);
    for s in &skills {
        names.insert(s.name.clone());
    }
    let custom = "custom-skill".to_string();
    let debugger = "debugger".to_string();
    let code_review = "code-review".to_string();
    let refactor = "refactor".to_string();
    assert!(names.contains(&custom));
    assert!(!names.contains(&debugger));
    assert!(!names.contains(&code_review));
    assert!(!names.contains(&refactor));

    match old_home {
        Some(v) => unsafe {
            std::env::set_var("HOME", v);
        },
        None => unsafe {
            std::env::remove_var("HOME");
        },
    }
    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn test_simple_format_parsing() {
    let content = r#"---
name: test-skill
description: test skill for helping users
tools:
  - read_file
  - write_file
priority: 50
---

test prompt"#;

    let skill = parse_skill_front_matter(content).unwrap();
    assert_eq!(skill.name, "test-skill");
    assert_eq!(skill.description, "test skill for helping users");
    assert_eq!(skill.tools, vec!["read_file", "write_file"]);
    assert_eq!(skill.priority, 50);
}

#[test]
fn parse_skill_front_matter_ignores_legacy_triggers_field() {
    let content = r#"---
name: test-skill
description: test skill for helping users
triggers:
  - exact phrase
  - another trigger
tools:
  - read_file
---

test prompt"#;

    let skill = parse_skill_front_matter(content).unwrap();
    assert_eq!(skill.name, "test-skill");
    assert_eq!(skill.description, "test skill for helping users");
    assert_eq!(skill.tools, vec!["read_file"]);
}

#[test]
fn load_skills_from_dir_supports_package_directory() {
    let dir = std::env::temp_dir().join(format!("rust-tools-skills-{}", uuid::Uuid::new_v4()));
    let package_dir = dir.join("superpower");
    std::fs::create_dir_all(package_dir.join("references")).unwrap();
    std::fs::write(
        package_dir.join("SKILL.md"),
        r#"---
name: superpower
description: packaged skill
---

Read references/guide.md before acting."#,
    )
    .unwrap();
    std::fs::write(package_dir.join("references").join("guide.md"), "resource").unwrap();

    let skills = load_skills_from_dir(&dir);
    let skill = skills.iter().find(|s| s.name == "superpower").unwrap();
    assert_eq!(skill.description, "packaged skill");
    assert_eq!(
        skill.resource_path.as_deref(),
        Some(package_dir.display().to_string().as_str())
    );
    assert!(skill.source_path.as_deref().unwrap().ends_with("SKILL.md"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_skills_from_dir_supports_zip_package_with_wrapped_root() {
    let dir = std::env::temp_dir().join(format!("rust-tools-skills-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let zip_path = dir.join("superpower.zip");
    write_zip_package(
        &zip_path,
        &[
            (
                "superpower/SKILL.md",
                r#"---
name: superpower
description: zipped skill
priority: 9
---

Use bundled references."#,
            ),
            ("superpower/references/guide.md", "resource"),
        ],
    );

    let skills = load_skills_from_dir(&dir);
    let skill = skills.iter().find(|s| s.name == "superpower").unwrap();
    let resource_path = PathBuf::from(skill.resource_path.as_deref().unwrap());
    assert_eq!(skill.description, "zipped skill");
    assert_eq!(skill.priority, 9);
    assert!(skill.source_path.as_deref().unwrap().contains(".zip!"));
    assert!(resource_path.join("references").join("guide.md").is_file());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_all_skills_discovers_external_installed_skill_packages() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let home = std::env::temp_dir().join(format!("rust-tools-home-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&home).unwrap();
    let old_home = std::env::var("HOME").ok();
    unsafe {
        std::env::set_var("HOME", &home);
    }

    let trae_builtin = home.join(".trae-cn/builtin/global/skills/web-dev");
    let trae_user = home.join(".trae-cn/skills/bits-code-guard");
    let trae_extension = home.join(".trae-cn/extensions/pylance/skills/pylance-refactoring");
    for (dir, name, description) in [
        (&trae_builtin, "web-dev", "web skill"),
        (&trae_user, "bits-code-guard", "review skill"),
        (&trae_extension, "pylance-refactoring", "refactoring skill"),
    ] {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            dir.join("SKILL.md"),
            format!("---\nname: {name}\ndescription: {description}\n---\n\nbody\n"),
        )
        .unwrap();
    }

    let skills = load_all_skills();
    let names = skills.iter().map(|s| s.name.as_str()).collect::<Vec<_>>();
    assert!(names.contains(&"web-dev"));
    assert!(names.contains(&"bits-code-guard"));
    assert!(names.contains(&"pylance-refactoring"));
    let web_dev = skills.iter().find(|s| s.name == "web-dev").unwrap();
    assert_eq!(
        web_dev.resource_path.as_deref(),
        Some(trae_builtin.display().to_string().as_str())
    );
    assert!(
        web_dev
            .source_path
            .as_deref()
            .unwrap()
            .ends_with("SKILL.md")
    );

    match old_home {
        Some(v) => unsafe {
            std::env::set_var("HOME", v);
        },
        None => unsafe {
            std::env::remove_var("HOME");
        },
    }
    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn skill_watch_roots_only_include_existing_skill_containers() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let home = std::env::temp_dir().join(format!("rust-tools-home-{}", uuid::Uuid::new_v4()));
    let old_home = std::env::var("HOME").ok();
    unsafe {
        std::env::set_var("HOME", &home);
    }

    let trae_builtin = home.join(".trae-cn/builtin_skills");
    let trae_user = home.join(".trae-cn/skills");
    let trae_nested = home.join(".trae-cn/builtin/global/skills");
    let trae_extension = home.join(".trae-cn/extensions/pylance/skills");
    let unrelated = home.join(".trae-cn/workspaces/project/cache");
    for dir in [
        &trae_builtin,
        &trae_user,
        &trae_nested,
        &trae_extension,
        &unrelated,
    ] {
        std::fs::create_dir_all(dir).unwrap();
    }

    let roots = skill_watch_roots().into_iter().collect::<BTreeSet<_>>();
    for expected in [skills_dir(), trae_builtin, trae_user, trae_nested, trae_extension] {
        assert!(roots.contains(&expected), "missing root {}", expected.display());
    }
    assert!(!roots.contains(&home.join(".trae-cn")));
    assert!(!roots.contains(&unrelated));

    match old_home {
        Some(value) => unsafe {
            std::env::set_var("HOME", value);
        },
        None => unsafe {
            std::env::remove_var("HOME");
        },
    }
    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn load_skills_from_dir_supports_collection_directory() {
    // feishu 式集合：collection/skills/<pkg>/SKILL.md，无根 manifest。
    let dir = std::env::temp_dir().join(format!("rust-tools-skills-{}", uuid::Uuid::new_v4()));
    let collection = dir.join("feishu");
    for (pkg, desc) in [("lark-base", "base skill"), ("lark-im", "im skill")] {
        let pkg_dir = collection.join("skills").join(pkg);
        std::fs::create_dir_all(pkg_dir.join("references")).unwrap();
        std::fs::write(
            pkg_dir.join("SKILL.md"),
            format!("---\nname: {pkg}\ndescription: {desc}\n---\n\nbody\n"),
        )
        .unwrap();
        std::fs::write(pkg_dir.join("references").join("guide.md"), "resource").unwrap();
    }

    let skills = load_skills_from_dir(&dir);
    let base = skills.iter().find(|s| s.name == "lark-base").unwrap();
    let im = skills.iter().find(|s| s.name == "lark-im").unwrap();
    assert_eq!(base.description, "base skill");
    assert_eq!(im.description, "im skill");
    assert_eq!(
        base.resource_path.as_deref(),
        Some(
            collection
                .join("skills")
                .join("lark-base")
                .display()
                .to_string()
                .as_str()
        )
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_skills_from_dir_single_package_dir_does_not_descend() {
    // 单包目录内若有 references/*.skill 资源，不能被误判为额外 skill。
    let dir = std::env::temp_dir().join(format!("rust-tools-skills-{}", uuid::Uuid::new_v4()));
    let package_dir = dir.join("argos-tools");
    std::fs::create_dir_all(package_dir.join("reference")).unwrap();
    std::fs::write(
        package_dir.join("SKILL.md"),
        "---\nname: argos-tools\ndescription: single package\n---\n\nbody\n",
    )
    .unwrap();
    std::fs::write(
        package_dir.join("reference").join("install.skill"),
        "---\nname: bogus-nested\ndescription: should-not-load\n---\n\nnope\n",
    )
    .unwrap();

    let skills = load_skills_from_dir(&dir);
    assert_eq!(skills.iter().filter(|s| s.name == "argos-tools").count(), 1);
    assert!(skills.iter().all(|s| s.name != "bogus-nested"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_skills_from_dir_supports_multi_package_collection_zip() {
    let dir = std::env::temp_dir().join(format!("rust-tools-skills-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let zip_path = dir.join("feishu.zip");
    write_zip_package(
        &zip_path,
        &[
            (
                "feishu/skills/lark-base/SKILL.md",
                "---\nname: lark-base\ndescription: base skill\n---\n\nbody\n",
            ),
            ("feishu/skills/lark-base/references/guide.md", "resource"),
            (
                "feishu/skills/lark-im/SKILL.md",
                "---\nname: lark-im\ndescription: im skill\n---\n\nbody\n",
            ),
        ],
    );

    let skills = load_skills_from_dir(&dir);
    let base = skills.iter().find(|s| s.name == "lark-base").unwrap();
    let im = skills.iter().find(|s| s.name == "lark-im").unwrap();
    assert_eq!(base.description, "base skill");
    assert_eq!(im.description, "im skill");
    assert!(base.source_path.as_deref().unwrap().contains(".zip!"));
    let resource_path = PathBuf::from(base.resource_path.as_deref().unwrap());
    assert!(resource_path.join("references").join("guide.md").is_file());
    let _ = std::fs::remove_dir_all(&dir);
}
