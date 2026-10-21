# Sign2Text QA Test Report

**Date:** 2026-10-16  
**Tester:** Automated QA (Cursor agent)  
**Build:** `assembleDebug` + `testDebugUnitTest`  
**Device under test:** Android Emulator `Small_Phone` (x86_64 / API 37)

---

## Summary

| Area | Result |
|------|--------|
| Gradle debug build | **PASS** |
| Unit tests (8) | **PASS** |
| Model assets packaged in APK | **PASS** |
| Welcome screen launch | **PASS** |
| Save / Delete / Confirm UI logic (unit) | **PASS** |
| Main camera screen on **x86_64 emulator** | **FAIL** (native MediaPipe lib missing for x86_64) |
| Live ASL recognition E2E | **BLOCKED** (needs ARM device or x86 32-bit emulator) |

**Verdict:** Core app build and spelling Save/Delete logic are good. Full on-device recognition must be verified on a **physical Android phone (arm64)** — not this x86_64 emulator.

---

## 1. Build & unit tests

| Check | Result | Notes |
|-------|--------|-------|
| `./gradlew assembleDebug` | PASS | `BUILD SUCCESSFUL` |
| `./gradlew testDebugUnitTest` | PASS | 8 tests, 0 failures |
| `LetterSequenceAccumulatorTest` | PASS | 6 tests including Save→next word→clear scenario |
| `WordDictionaryTest` | PASS | 2 tests |

### Save / Delete scenario covered by unit test

`saveThenSpellNextWordThenClear`:

1. Spell `HI` → take/clear (Save) → buffer empty, saved value `HI`
2. Spell `BYE` → buffer `BYE`
3. `clear()` (Delete) → buffer empty

---

## 2. APK assets (required for recognition)

| Asset | In APK? |
|-------|---------|
| `asl_mlp_fp32.tflite` | YES |
| `hand_landmarker.task` | YES |
| `label_map.json` | YES |
| `feature_norm.json` | YES |
| `calibration.json` | YES |
| `words_en.txt` | YES |

---

## 3. UI / instrumentation on emulator

| Step | Result | Notes |
|------|--------|-------|
| Install APK | PASS | `adb install -r` Success |
| Launch `WelcomeActivity` | PASS | Brand + **Let's start** visible |
| 16 KB page-size warning dialog | INFO | Emulator shows compatibility warning (TFLite/CameraX libs); dismiss with **Don't Show Again** |
| Tap **Let's start** → `MainActivity` | FAIL | App crashes immediately |

### Crash root cause (emulator only)

```
java.lang.UnsatisfiedLinkError:
  dlopen failed: library "libmediapipe_tasks_vision_jni.so" not found
  at HandLandmarker.<clinit>
  at MainActivity.createHandLandmarker
  at MainActivity.initializePipeline
```

**APK native libraries present:**

| ABI | MediaPipe JNI | TFLite JNI |
|-----|---------------|------------|
| arm64-v8a | YES | YES |
| armeabi-v7a | YES | YES |
| x86 | YES | YES |
| **x86_64** | **NO** | YES |

`Small_Phone` is **x86_64**, so MediaPipe cannot load → Main screen crashes.  
A real **arm64 phone** includes the MediaPipe `.so` and should open Main normally.

---

## 4. New Save / Delete feature — expected behavior (manual on phone)

Test on a physical device after opening **Spell words**:

| ID | Steps | Expected |
|----|-------|----------|
| S1 | Spell H then I → tap **Save** | **Saved** shows `HI`; Spelling clears |
| S2 | Spell B Y E → tap **Save** | **Saved** shows `BYE` |
| S3 | Tap **Delete** | Spelling + Saved cleared; message “Cleared…” |
| S4 | Tap **Save** with empty spelling | “Nothing spelled yet.” |
| S5 | Spell a word → **Confirm** | Dictionary match + Saved filled |
| S6 | No crash after Save/Delete/Confirm | App stays on Main |

---

## 5. Recommended next QA (manual)

1. Connect a **physical Android phone** (API 26+).
2. In Android Studio: Run `app` on that device (not x86_64 emulator).
3. Execute table in §4.
4. Optionally create an **x86** (32-bit) AVD if you must use emulator — MediaPipe ships `lib/x86/`.

---

## 6. Defects logged

| ID | Severity | Description | Status |
|----|----------|-------------|--------|
| DEF-01 | High (emulator) / N/A (ARM phone) | MainActivity crashes on x86_64 emulator: MediaPipe JNI not packaged for x86_64 | Known dependency limitation |
| DEF-02 | Low | Emulator shows Android “16 KB page size” compatibility dialog on first launch | Informational |

No defects found in Save/Delete spelling buffer logic under unit test.

---

## 7. Conclusion

- **Build + unit tests + assets:** project is healthy.  
- **Save/Delete feature:** logic verified by unit test; UI needs confirmation on a **real phone**.  
- **Emulator camera path:** blocked by missing MediaPipe `x86_64` native library — use a physical device for full sign-to-text QA.
