#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGURATION="release"
PRODUCT_NAME="NStar"
PRODUCT_EXECUTABLE="nstar-gui"
OUTPUT_DIR="$SCRIPT_DIR/dist-swift"
BUNDLE_ID="com.nstar.gui"
MIN_MACOS="10.13"
BACKEND_NAME="nstar-backend"
BACKEND_SOURCE_MODE=0
PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/.venv/bin/python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug)
      CONFIGURATION="debug"
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --product-name)
      PRODUCT_NAME="$2"
      shift 2
      ;;
    --bundle-id)
      BUNDLE_ID="$2"
      shift 2
      ;;
    --min-macos)
      MIN_MACOS="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --source-backend)
      BACKEND_SOURCE_MODE=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

echo "[build] Compiling Swift GUI ($CONFIGURATION)"
mkdir -p "$SCRIPT_DIR/.build/clang-module-cache" "$SCRIPT_DIR/.build/swiftpm-module-cache"
CLANG_MODULE_CACHE_PATH="$SCRIPT_DIR/.build/clang-module-cache" \
SWIFTPM_MODULECACHE_OVERRIDE="$SCRIPT_DIR/.build/swiftpm-module-cache" \
swift build --disable-sandbox -c "$CONFIGURATION" --product "$PRODUCT_EXECUTABLE"

APP_DIR="$OUTPUT_DIR/$PRODUCT_NAME.app"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"
BACKEND_DIR="$RESOURCES_DIR/backend"
BACKEND_BUILD_DIR="$SCRIPT_DIR/.build/pyinstaller/backend/dist"
BACKEND_WORK_DIR="$SCRIPT_DIR/.build/pyinstaller/backend/work"
BACKEND_SPEC_DIR="$SCRIPT_DIR/.build/pyinstaller/backend/spec"
BINARY_PATH="$SCRIPT_DIR/.build/$CONFIGURATION/$PRODUCT_EXECUTABLE"

if [[ ! -x "$BINARY_PATH" ]]; then
  echo "Built executable not found at $BINARY_PATH" >&2
  exit 1
fi

echo "[build] Assembling app bundle at $APP_DIR"
rm -rf "$APP_DIR"
mkdir -p "$MACOS_DIR" "$BACKEND_DIR"

cp "$BINARY_PATH" "$MACOS_DIR/$PRODUCT_NAME"
chmod +x "$MACOS_DIR/$PRODUCT_NAME"

if [[ "$BACKEND_SOURCE_MODE" -eq 1 ]]; then
  echo "[build] Copying Python source backend"
  cp -R "$SCRIPT_DIR/nstar" "$BACKEND_DIR/"
else
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python runtime for backend build not found at $PYTHON_BIN" >&2
    exit 1
  fi

  echo "[build] Freezing Python backend with PyInstaller"
  rm -rf "$BACKEND_BUILD_DIR"
  "$PYTHON_BIN" "$SCRIPT_DIR/nstar/build_macos_executable.py" \
    --name "$BACKEND_NAME" \
    --entrypoint "$SCRIPT_DIR/nstar/gui_bridge.py" \
    --dist-dir "$BACKEND_BUILD_DIR" \
    --work-dir "$BACKEND_WORK_DIR" \
    --spec-dir "$BACKEND_SPEC_DIR" \
    --onedir \
    --clean

  cp -R "$BACKEND_BUILD_DIR/$BACKEND_NAME" "$BACKEND_DIR/$BACKEND_NAME"
fi

cat > "$CONTENTS_DIR/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleExecutable</key>
  <string>$PRODUCT_NAME</string>
  <key>CFBundleIdentifier</key>
  <string>$BUNDLE_ID</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>$PRODUCT_NAME</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>1.0</string>
  <key>CFBundleVersion</key>
  <string>1</string>
  <key>LSMinimumSystemVersion</key>
  <string>$MIN_MACOS</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
EOF

echo "[build] App bundle ready at $APP_DIR"
if [[ "$BACKEND_SOURCE_MODE" -eq 1 ]]; then
  echo "[build] Backend copied to $BACKEND_DIR/nstar"
  echo "[build] Source backend mode still requires a Python runtime with torch and midiutil installed."
else
  echo "[build] Frozen backend copied to $BACKEND_DIR/$BACKEND_NAME/$BACKEND_NAME"
  echo "[build] Runtime internet access is not required by the app bundle."
fi
