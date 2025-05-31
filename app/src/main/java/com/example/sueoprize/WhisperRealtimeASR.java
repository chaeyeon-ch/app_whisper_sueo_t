package com.example.sueoprize;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import androidx.core.app.ActivityCompat;

import com.example.sueoprize.engine.WhisperEngine;
import com.example.sueoprize.engine.WhisperEngineJava;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.atomic.AtomicBoolean;

public class WhisperRealtimeASR {

    public interface Listener {
        void onResult(String result);
        void onError(String error);
    }

    private final Context context;
    private final WhisperEngine engine;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private Listener listener;

    private AudioRecord recorder;
    private Thread recordThread;
    private final AtomicBoolean isRecording = new AtomicBoolean(false);

    public WhisperRealtimeASR(Context context) {
        this.context = context;
        this.engine = new WhisperEngineJava(context);
    }

    public void setListener(Listener listener) {
        this.listener = listener;
    }

    public boolean isInitialized() {
        return engine.isInitialized();
    }
    public boolean isRecording() {
        return isRecording.get();
    }

    public void loadModelFromAssets(String modelAssetName, String vocabAssetName) {
        try {
            String modelPath = copyAssetToFile(context, modelAssetName);
            String vocabPath = copyAssetToFile(context, vocabAssetName);
            engine.initialize(modelPath, vocabPath, true);
        } catch (Exception e) {
            if (listener != null) listener.onError("Model load failed: " + e.getMessage());
        }
    }


    public void startListening() {
        if (isRecording.get()) return;

        if (ActivityCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            if (listener != null) listener.onError("RECORD_AUDIO permission not granted");
            return;
        }

        int sampleRate = 16000;
        int bufferSize = AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        recorder = new AudioRecord(MediaRecorder.AudioSource.MIC, sampleRate,
                AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufferSize);

        isRecording.set(true);
        recorder.startRecording();

        recordThread = new Thread(() -> {
            ByteBuffer buffer = ByteBuffer.allocateDirect(sampleRate * 2 * 3); // 3초 버퍼
            buffer.order(ByteOrder.nativeOrder());
            byte[] readBuffer = new byte[bufferSize];

            while (isRecording.get() && buffer.position() < buffer.capacity()) {
                int read = recorder.read(readBuffer, 0, bufferSize);
                if (read > 0) buffer.put(readBuffer, 0, read);
            }

            recorder.stop();
            recorder.release();
            isRecording.set(false);

            float[] samples = convertToFloatArray(buffer);
            String result = engine.transcribeBuffer(samples);

            mainHandler.post(() -> {
                if (listener != null) listener.onResult(result);
            });
        });
        recordThread.start();
    }

    public void stopListening() {
        isRecording.set(false);
    }

    private float[] convertToFloatArray(ByteBuffer buffer) {
        buffer.flip();
        float[] samples = new float[buffer.remaining() / 2];
        for (int i = 0; i < samples.length; i++) {
            samples[i] = buffer.getShort() / 32768.0f;
        }
        return samples;
    }

    private String copyAssetToFile(Context context, String assetName) throws IOException {
        java.io.File outFile = new java.io.File(context.getFilesDir(), assetName);
        if (outFile.exists()) return outFile.getAbsolutePath();

        try (java.io.InputStream is = context.getAssets().open(assetName);
             java.io.FileOutputStream fos = new java.io.FileOutputStream(outFile)) {
            byte[] buffer = new byte[1024];
            int length;
            while ((length = is.read(buffer)) > 0) {
                fos.write(buffer, 0, length);
            }
        }
        return outFile.getAbsolutePath();
    }

}
