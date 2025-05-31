package com.example.sueoprize

import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.example.sueoprize.WhisperRealtimeASR
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var textViewTranslation: TextView
    private lateinit var textViewSpeechRecognition: TextView
    private lateinit var btnRecord: Button
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var whisperASR: WhisperRealtimeASR

    private val TAG = "MainActivity"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        textViewTranslation = findViewById(R.id.textViewTranslation)
        textViewSpeechRecognition = findViewById(R.id.textViewSpeechRecognition)
        btnRecord = findViewById(R.id.btnRecord)

        cameraExecutor = Executors.newSingleThreadExecutor()

        // Whisper ASR 초기화
        whisperASR = WhisperRealtimeASR(this)
        whisperASR.setListener(object : WhisperRealtimeASR.Listener {
            override fun onResult(result: String) {
                textViewSpeechRecognition.text = result
            }

            override fun onError(error: String) {
                textViewSpeechRecognition.text = "오류: $error"
            }
        })

        whisperASR.loadModelFromAssets("whisper-tiny.tflite", "filters_vocab_multilingual.bin")


        btnRecord.setOnClickListener {
            if (whisperASR.isInitialized() && !whisperASR.isRecording) {
                whisperASR.startListening()
                btnRecord.text = "정지"
            } else {
                whisperASR.stopListening()
                btnRecord.text = "음성 인식 시작"
            }
        }

        startCamera()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, HandAnalyzer(this, textViewTranslation))
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalysis
                )
            } catch (e: Exception) {
                Log.e(TAG, "Camera binding failed", e)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        whisperASR.stopListening()
    }
}