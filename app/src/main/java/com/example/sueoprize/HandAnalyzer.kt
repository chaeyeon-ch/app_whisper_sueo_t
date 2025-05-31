package com.example.sueoprize

import android.app.Activity
import android.content.Context
import android.util.Log
import android.widget.TextView
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import kotlin.math.acos
import kotlin.math.sqrt

class HandAnalyzer(
    private val context: Context,
    private val textView: TextView
) : ImageAnalysis.Analyzer {

    private val handLandmarker: HandLandmarker
    private val classifier = TFLiteSignClassifier(context)
    private val inputSequence = ArrayList<FloatArray>()
    private val REQUIRED_SEQUENCE_LENGTH = 10
    private val labels: List<String> = loadLabelsFromJson(context, "labels.json")

    private var hasHandPreviously = false
    private var readyToPredict = true

    init {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("hand_landmarker.task")
            .build()

        val options = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setResultListener { result: HandLandmarkerResult?, _ ->

                val landmarksList = result?.landmarks()

                Log.d("HandDebug", "hasHandPreviously=$hasHandPreviously, readyToPredict=$readyToPredict, landmarks.isNullOrEmpty=${landmarksList.isNullOrEmpty()}")

                if (landmarksList.isNullOrEmpty()) {
                    hasHandPreviously = false
                    readyToPredict = true
                    return@setResultListener
                }

                // 손 처음 등장 시 시퀀스 초기화
                if (!hasHandPreviously && readyToPredict) {
                    hasHandPreviously = true
                    readyToPredict = false
                    inputSequence.clear()
                    Log.d("PredictReady", "✋ 손 등장! 시퀀스 수집 시작")
                    return@setResultListener
                }

                if (hasHandPreviously && !landmarksList.isNullOrEmpty()) {
                    landmarksList.firstOrNull()?.let { landmarks ->
                        val features = extractFeatures(landmarks)
                        inputSequence.add(features)
                        Log.d("Sequence", "시퀀스 수집 중: ${inputSequence.size}/$REQUIRED_SEQUENCE_LENGTH")

                        if (inputSequence.size == REQUIRED_SEQUENCE_LENGTH) {
                            val batchInput = Array(1) { Array(REQUIRED_SEQUENCE_LENGTH) { FloatArray(56) } }
                            for (i in inputSequence.indices) {
                                batchInput[0][i] = inputSequence[i]
                            }

                            val prediction = classifier.predict(batchInput)
                            val predictedLabel = labels.getOrNull(prediction) ?: "?"

                            (context as? Activity)?.runOnUiThread {
                                val prevText = textView.text.toString()
                                textView.text = prevText + predictedLabel
                                Log.d("Prediction", "✅ 예측 완료: $predictedLabel → ${prevText + predictedLabel}")
                            }

                            inputSequence.clear()
                        }
                    }
                }
            }
            .build()

        handLandmarker = HandLandmarker.createFromOptions(context, options)
    }

    override fun analyze(image: ImageProxy) {
        val bitmap = image.toBitmap() ?: run {
            image.close()
            return
        }
        val mpImage = BitmapImageBuilder(bitmap).build()
        handLandmarker.detectAsync(mpImage, image.imageInfo.timestamp)
        image.close()
    }

    private fun extractFeatures(landmarks: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>): FloatArray {
        val features = FloatArray(56)
        val jointPairs = listOf(
            0 to 1, 1 to 2, 2 to 3, 3 to 4,
            0 to 5, 5 to 6, 6 to 7, 7 to 8,
            5 to 9, 9 to 10, 10 to 11, 11 to 12,
            9 to 13, 13 to 14, 14 to 15, 15 to 16,
            13 to 17, 17 to 18, 18 to 19, 19 to 20
        )

        val vectors = jointPairs.map { (i, j) ->
            val x = landmarks[j].x() - landmarks[i].x()
            val y = landmarks[j].y() - landmarks[i].y()
            floatArrayOf(x, y)
        }

        for ((i, v) in vectors.withIndex()) {
            features[i * 2] = v[0]
            features[i * 2 + 1] = v[1]
        }

        val angleIndices = listOf(
            0 to 1, 1 to 2, 2 to 3,
            4 to 5, 5 to 6, 6 to 7,
            8 to 9, 9 to 10, 10 to 11,
            12 to 13, 13 to 14, 14 to 15,
            16 to 17, 17 to 18, 18 to 19
        )

        fun dot(u: FloatArray, v: FloatArray): Float = u[0] * v[0] + u[1] * v[1]
        fun norm(v: FloatArray): Float = sqrt(v[0] * v[0] + v[1] * v[1])

        for ((index, pair) in angleIndices.withIndex()) {
            val (i, j) = pair
            val v1 = vectors[i]
            val v2 = vectors[j]
            val cosTheta = (dot(v1, v2) / (norm(v1) * norm(v2) + 1e-6f)).coerceIn(-1f, 1f)
            val angle = acos(cosTheta.toDouble()).toFloat()
            features[40 + index] = angle
        }

        return features
    }

    private fun loadLabelsFromJson(context: Context, filename: String): List<String> {
        return try {
            val inputStream = context.assets.open(filename)
            val bufferedReader = BufferedReader(InputStreamReader(inputStream))
            val jsonString = bufferedReader.use { it.readText() }
            val jsonObject = JSONObject(jsonString)
            val jsonArray = jsonObject.getJSONArray("labels")
            List(jsonArray.length()) { i -> jsonArray.getString(i) }
        } catch (e: Exception) {
            e.printStackTrace()
            emptyList()
        }
    }
}
