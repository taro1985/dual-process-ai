#!/usr/bin/env bash
set -e

HOOK_PATH=".git/hooks/pre-push"

echo "🔧 Installing DPAI Git Pre-Push Guard..."

cat << 'EOF' > "$HOOK_PATH"
#!/usr/bin/env bash
#
# DPAI Dual-Remote Guard: Pre-Push Hook
# Enforces:
# 1. 'origin' (Main/Release repo) is LOCKED until explicit user approval (ALLOW_PUBLIC_PUSH=1).
# 2. 'agi-internal' must NEVER be pushed to 'origin'.
# 3. 'private' (dual-process-ai-internal) and 'vaio' allow full push for internal development.

remote="$1"
url="$2"
current_branch=$(git symbolic-ref --short HEAD 2>/dev/null)

while read local_ref local_oid remote_ref remote_oid
do
    # If pushing to origin (public release repo)
    if [[ "$remote" == "origin" || "$url" == *"dual-process-ai.git" ]]; then
        # 1. Block agi-internal branch completely from origin
        if [[ "$current_branch" == "agi-internal" || "$local_ref" == *"agi-internal"* ]]; then
            echo "════════════════════════════════════════════════════════════════════════" >&2
            echo "🚫 [DPAI Guard] PUSH TO ORIGIN BLOCKED!" >&2
            echo "   Branch '$current_branch' contains internal AGI/System 3 experimental data." >&2
            echo "   It must NEVER be pushed to 'origin'." >&2
            echo "   Please push to 'private' instead: git push private $current_branch" >&2
            echo "════════════════════════════════════════════════════════════════════════" >&2
            exit 1
        fi

        # 2. Lock origin from all pushes until explicit ALLOW_PUBLIC_PUSH=1
        if [[ "$ALLOW_PUBLIC_PUSH" != "1" ]]; then
            echo "════════════════════════════════════════════════════════════════════════" >&2
            echo "🔒 [DPAI Guard] ORIGIN REPOSITORY IS LOCKED!" >&2
            echo "   Per user policy: 'origin' is locked until explicit publish instruction." >&2
            echo "   Use 'private' remote for all day-to-day development:" >&2
            echo "     git push private $current_branch" >&2
            echo "   To force push to origin when authorized, run:" >&2
            echo "     ALLOW_PUBLIC_PUSH=1 git push origin $current_branch" >&2
            echo "════════════════════════════════════════════════════════════════════════" >&2
            exit 1
        fi
    fi
done

exit 0
EOF

chmod +x "$HOOK_PATH"
echo "✅ Pre-push hook installed successfully at $HOOK_PATH"
