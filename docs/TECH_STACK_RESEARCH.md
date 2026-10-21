# System Architecture and Technology Stack

**(Research Paper–Ready Explanation)**

## 1. Overview of the System Architecture

Sign2Text is a **mobile-based American Sign Language (ASL) alphabet recognition system** that converts static hand signs (A–Z) into text in real time. Unlike conventional client–server applications, the system adopts an **on-device processing architecture**. All computation—from camera capture to classification—is performed locally on the Android device.

This design choice is intentional and significant for the research contribution:

- **Privacy:** Hand images and landmark data never leave the device.
- **Low latency:** Recognition does not depend on network round-trips.
- **Offline usability:** The application functions without internet connectivity.
- **Deployability:** A single Android APK contains both the user interface and the inference pipeline.

In terms of classical software layering, the project can be described as follows:

| Layer | Role in Sign2Text |
|--------|-------------------|
| **Frontend** | Android user interface and camera interaction |
| **Backend / Processing Layer** | On-device ML inference pipeline (not a remote server) |
| **Model Development Layer** | Offline Python training and export toolchain |

---

## 2. Frontend

### 2.1 Platform and Language

The frontend is implemented as a **native Android application** using **Kotlin**. Native development was selected to ensure:

- direct access to camera hardware,
- efficient integration with on-device ML libraries, and
- stable real-time performance suitable for continuous frame processing.

The application targets **Android API level 26 and above**, with compile/target SDK 34.

### 2.2 User Interface Framework

The interface is built using:

- **XML layouts** for screen structure, and
- **Material Components (Material Design)** for UI controls and visual consistency.

There are two primary user-facing screens:

1. **Welcome Screen (`WelcomeActivity`)**  
   Serves as the entry point of the application. It presents branding, a brief introduction, credits, and a call-to-action to begin recognition.

2. **Recognition Screen (`MainActivity`)**  
   Contains the live camera preview, predicted letter display, status messages, and optional spell-words mode controls.

Navigation is activity-based (explicit Android Intents). The application does not use a web frontend, React Native, Flutter, or a multi-page navigation graph.

### 2.3 Camera Interface

Live visual input is obtained through **CameraX**, Google’s modern Android camera library. CameraX provides:

- preview rendering for the user, and
- continuous image analysis frames for the recognition pipeline.

This frontend layer is responsible only for **capturing interaction and presenting results**. It does not perform classification itself; it forwards frames to the on-device processing layer and displays returned predictions.

### 2.4 Frontend Responsibilities (Summary)

- Request and handle camera permission
- Display live preview
- Show predicted letter / uncertainty state
- Support letter-only and spell-words interaction modes
- Present dictionary match feedback and system status messages

---

## 3. Backend / Processing Layer (On-Device Inference)

### 3.1 Clarification for Academic Context

In traditional web systems, “backend” refers to a remote server, API, and database. **Sign2Text does not use a remote backend.**

Instead, the equivalent computational backend is an **embedded, on-device inference pipeline** packaged inside the Android application. For research documentation, this may be accurately described as:

> *an on-device machine learning backend*  
> or  
> *a local inference engine*

This distinction should be stated clearly in the paper to avoid misunderstanding.

### 3.2 Processing Pipeline

The backend pipeline executes the following stages for each camera frame:

1. **Frame Acquisition** – CameraX supplies image frames.
2. **Hand Landmark Detection** – MediaPipe Hand Landmarker extracts hand keypoints.
3. **Feature Extraction and Normalization** – Landmarks are converted into a numeric feature vector and normalized.
4. **Classification** – A TensorFlow Lite MLP model predicts an ASL letter (A–Z).
5. **Temporal Stabilization** – Smoothing, majority voting, and debouncing reduce flicker.
6. **Optional Word Assembly** – Stable letters may be accumulated and matched against a local dictionary.
7. **UI Update** – Final letter/word output is returned to the frontend.

### 3.3 Why There Is No Cloud Backend

The recognition model and supporting assets are stored locally in the application’s `assets` directory, including:

- MediaPipe hand landmarker model (`.task`)
- TensorFlow Lite classifier (`.tflite`)
- Label map, feature normalization parameters, calibration data
- Optional English word dictionary

Because inference is fully local, the system requires **no REST API, no authentication server, and no remote database** during runtime.

---

## 4. Libraries and Frameworks

### 4.1 Android Application Libraries

| Library | Purpose in the System |
|---------|------------------------|
| **Kotlin / Android SDK** | Core application logic and activity lifecycle |
| **AndroidX AppCompat & Core KTX** | Compatibility and modern Android utilities |
| **Material Components** | UI components and interaction design |
| **CameraX** | Camera preview and real-time frame analysis |
| **MediaPipe Tasks Vision** | Hand landmark detection from video frames |
| **TensorFlow Lite** | On-device neural network inference for letter classification |

#### Justification of Key Libraries

**CameraX**  
Enables reliable real-time camera access with lifecycle awareness, which is essential for continuous ASL recognition.

**MediaPipe Hand Landmarker**  
Detects 21 hand landmarks per frame. Landmark-based representation reduces input dimensionality compared with raw image classification and improves robustness for gesture-related tasks.

**TensorFlow Lite**  
Allows deployment of a trained Multi-Layer Perceptron (MLP) classifier directly on mobile hardware with low latency and no network dependency.

### 4.2 Offline Model Development Libraries (Python Toolchain)

Although not part of the runtime mobile app, the research workflow includes a Python-based training and export pipeline:

| Library | Purpose |
|---------|---------|
| **TensorFlow** | Train the MLP model and export SavedModel/TFLite |
| **MediaPipe (Python)** | Extract landmarks from dataset images |
| **OpenCV** | Image handling during dataset preprocessing |
| **NumPy / Pandas** | Numerical computation and dataset management |
| **scikit-learn** | Data splitting and evaluation support |

This toolchain converts raw ASL alphabet images into landmark features, trains the classifier, calibrates outputs, exports a `.tflite` model, and synchronizes the model bundle into Android assets.

For the paper, this can be framed as the **offline training pipeline**, while the Android app is the **online/real-time deployment system**.

---

## 5. End-to-End Data Flow

The complete system flow may be summarized as:

**Offline phase:**  
Dataset images → landmark extraction → feature normalization → MLP training → TFLite export → packaging into Android assets.

**Online phase (mobile runtime):**  
User hand sign → camera frame → MediaPipe landmarks → normalized features → TFLite classification → stabilized letter prediction → text display (and optional word matching).

This separation of **training (offline)** and **inference (on-device)** is a standard and academically accepted mobile ML deployment pattern.

---

## 6. Suggested Wording for the Research Paper

> The Sign2Text system is implemented as a native Android application developed in Kotlin. The frontend consists of Material Design–based XML interfaces that provide onboarding and real-time recognition views, with CameraX used for live video capture. Unlike conventional cloud-dependent architectures, Sign2Text employs an on-device machine learning backend. Hand landmarks are extracted using MediaPipe Hand Landmarker and classified by a TensorFlow Lite Multi-Layer Perceptron model embedded within the application. Model training and export are performed offline using a Python-based pipeline with TensorFlow, MediaPipe, OpenCV, NumPy, Pandas, and scikit-learn. This architecture enables real-time, privacy-preserving, and offline ASL alphabet recognition without requiring a remote server.

---

## 7. One-Line Stack Summary

| Layer | Technologies |
|--------|----------------|
| **Frontend** | Native Android (Kotlin, XML, Material Components, CameraX) |
| **Backend** | On-device ML inference (MediaPipe + TensorFlow Lite); no remote server |
| **Training Stack** | Python (TensorFlow, MediaPipe, OpenCV, NumPy, Pandas, scikit-learn) |
