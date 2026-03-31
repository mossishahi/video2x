#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$SCRIPT_DIR/dist"
APP_NAME="Video2X"
APP_BUNDLE="$BUILD_DIR/$APP_NAME.app"
DMG_NAME="Video2X-macOS-arm64.dmg"

echo "=== Video2X macOS App Builder ==="
echo ""

# Step 1: Build video2x CLI if not already built
V2X_BIN="$REPO_DIR/build/video2x-install/bin/video2x"
if [ ! -f "$V2X_BIN" ]; then
    echo "[1/5] Building video2x CLI..."
    cd "$REPO_DIR"
    export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
    export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
    export OpenMP_ROOT=/opt/homebrew/opt/libomp
    export VULKAN_SDK=/opt/homebrew
    cmake -G Ninja -S . -B build \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=build/video2x-install \
        -DVIDEO2X_ENABLE_NATIVE=ON \
        -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
        -DOpenMP_CXX_LIB_NAMES="omp" \
        -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib \
        -DVulkan_INCLUDE_DIR=/opt/homebrew/include \
        -DVulkan_LIBRARY=/opt/homebrew/lib/libvulkan.dylib
    cmake --build build --config Release --parallel --target install
else
    echo "[1/5] video2x CLI already built, skipping."
fi

# Step 2: Set up Python venv
echo "[2/5] Setting up Python environment..."
cd "$SCRIPT_DIR"
if [ ! -d "venv" ]; then
    /opt/homebrew/bin/python3.12 -m venv venv
    source venv/bin/activate
    pip install pywebview flask py2app
else
    source venv/bin/activate
    pip install -q py2app 2>/dev/null || true
fi

# Step 3: Create .app bundle
echo "[3/5] Creating .app bundle..."
rm -rf "$BUILD_DIR"
mkdir -p "$APP_BUNDLE/Contents/MacOS"
mkdir -p "$APP_BUNDLE/Contents/Resources/bin"
mkdir -p "$APP_BUNDLE/Contents/Resources/lib"
mkdir -p "$APP_BUNDLE/Contents/Resources/models"

# Copy the Python app and venv
cp app.py "$APP_BUNDLE/Contents/Resources/"
cp -R venv "$APP_BUNDLE/Contents/Resources/venv"

# Copy video2x binary
cp "$V2X_BIN" "$APP_BUNDLE/Contents/Resources/bin/"

# Copy required dylibs
INSTALL_LIB="$REPO_DIR/build/video2x-install/lib"
for lib in "$INSTALL_LIB"/*.dylib; do
    [ -f "$lib" ] && cp "$lib" "$APP_BUNDLE/Contents/Resources/lib/"
done

# Copy key Homebrew dylibs
for lib in libvulkan.dylib libMoltenVK.dylib libomp.dylib libncnn.dylib; do
    SRC="/opt/homebrew/lib/$lib"
    [ -f "$SRC" ] && cp "$SRC" "$APP_BUNDLE/Contents/Resources/lib/"
done

# Copy models
cp -R "$REPO_DIR/build/video2x-install/share/video2x/models/"* "$APP_BUNDLE/Contents/Resources/models/"

# Copy app icon
if [ -f "$SCRIPT_DIR/AppIcon.icns" ]; then
    cp "$SCRIPT_DIR/AppIcon.icns" "$APP_BUNDLE/Contents/Resources/AppIcon.icns"
fi

# Copy MoltenVK ICD
mkdir -p "$APP_BUNDLE/Contents/Resources/vulkan/icd.d"
cp /opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json "$APP_BUNDLE/Contents/Resources/vulkan/icd.d/"

# Create launcher script
cat > "$APP_BUNDLE/Contents/MacOS/Video2X" << 'LAUNCHER'
#!/bin/bash
DIR="$(cd "$(dirname "$0")/../Resources" && pwd)"
export VK_ICD_FILENAMES="$DIR/vulkan/icd.d/MoltenVK_icd.json"
export DYLD_LIBRARY_PATH="$DIR/lib:/opt/homebrew/lib"
export PATH="$DIR/bin:$PATH"
source "$DIR/venv/bin/activate"
exec python "$DIR/app.py"
LAUNCHER
chmod +x "$APP_BUNDLE/Contents/MacOS/Video2X"

# Create Info.plist
cat > "$APP_BUNDLE/Contents/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Video2X</string>
    <key>CFBundleDisplayName</key>
    <string>Video2X</string>
    <key>CFBundleIdentifier</key>
    <string>com.video2x.macos</string>
    <key>CFBundleVersion</key>
    <string>6.4.0</string>
    <key>CFBundleShortVersionString</key>
    <string>6.4.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleExecutable</key>
    <string>Video2X</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.video</string>
</dict>
</plist>
PLIST

echo "[4/6] Ad-hoc signing .app bundle..."
codesign --force --deep --sign - "$APP_BUNDLE"

echo "[5/6] .app bundle created at: $APP_BUNDLE"

# Step 5: Create DMG
echo "[6/6] Creating DMG..."
DMG_PATH="$BUILD_DIR/$DMG_NAME"
DMG_TMP="$BUILD_DIR/dmg_tmp"
mkdir -p "$DMG_TMP"
cp -R "$APP_BUNDLE" "$DMG_TMP/"
ln -s /Applications "$DMG_TMP/Applications"

hdiutil create -volname "Video2X" -srcfolder "$DMG_TMP" -ov -format UDZO "$DMG_PATH"
rm -rf "$DMG_TMP"

echo ""
echo "=== Build Complete ==="
echo "App: $APP_BUNDLE"
echo "DMG: $DMG_PATH"
echo ""
echo "To install: Open the DMG and drag Video2X to Applications."
