#!/usr/bin/env bash
# Sourcing this script adds the project root and src/ to the PYTHONPATH
# allowing scripts in tool/ to be run from anywhere without import errors.

source venv/bin/activate

# Get the absolute path of the directory containing this script
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH}"

# Grant access to the SO-101 arm serial ports (resets on reboot/replug)
for port in /dev/ttyACM0 /dev/ttyACM1; do
    if [ -e "$port" ] && [ ! -w "$port" ]; then
        echo "=> Enabling access to ${port} (sudo may prompt for password)..."
        sudo chmod 666 "$port"
    fi
done

# Start the adb server and check the Meta Quest is authorized.
# The Quest shows as "unauthorized" until you put on the headset and
# accept the "Allow USB debugging" prompt (tick "Always allow").
if command -v adb &> /dev/null; then
    adb start-server &> /dev/null
    quest_status="$(adb devices | sed -n '2p' | awk '{print $2}')"
    case "$quest_status" in
        device)
            echo "✓ Meta Quest connected and authorized."
            ;;
        unauthorized)
            echo "⚠️ Meta Quest is connected but UNAUTHORIZED."
            echo "   Put on the headset and accept the 'Allow USB debugging' prompt."
            ;;
        *)
            echo "⚠️ No Meta Quest detected over USB (check the cable, or pass --ip-address)."
            ;;
    esac
else
    echo "⚠️ adb not installed — run install.sh (needed for Meta Quest teleop)."
fi

echo "✓ Environment ready. PYTHONPATH configured for Actoris Harena / VLA tools." # TODO: environment should be independent of Actoris Harena.
echo "  You can now run scripts from the tool/ directory."
