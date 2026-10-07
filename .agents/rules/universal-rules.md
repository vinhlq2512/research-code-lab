---
name: universal-rules
version: 1.0.0
priority: P0
trigger: always_on
---

# Universal Rules (TIER 0) - AG Kit

> Always-active rules that apply to every request, regardless of domain.

---

## 🌐 Language Handling

When user's prompt is NOT in English:

1. **Internally translate** for better comprehension
2. **Respond in user's language** - match their communication
3. **Code comments/variables** remain in English

---

## 🧹 Clean Code (Global Mandatory)

**ALL code MUST follow `@[skills/clean-code]` rules. No exceptions.**

- **Code**: Concise, direct, no over-engineering. Self-documenting.
- **Testing**: Mandatory. Pyramid (Unit > Int > E2E) + AAA Pattern.
- **Performance**: Measure first. Adhere to current Core Web Vitals standards.
- **Infra/Safety**: 5-Phase Deployment. Verify secrets security.

---

## 🛑 Git Commit Policy (Strict Mandatory)

- **NEVER auto-commit**: Do NOT execute `git commit` automatically or proactively after modifying code.
- **Explicit User Command Only**: ONLY execute `git commit` when the user explicitly requests or commands it (e.g. "commit cho tôi", "commit this", "hãy commit").
- **Inspect before committing**: When requested to commit, always verify which files should be committed and ensure temporary, secret, or unnecessary files are excluded/ignored.

---
