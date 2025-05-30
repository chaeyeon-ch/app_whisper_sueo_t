package com.example.sueoprize

import android.content.Context
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.flex.FlexDelegate
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.InputStreamReader
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class TFLiteSignClassifier(context: Context) {

    private var interpreter: Interpreter
    private var labels: List<String>

    init {
        val model = loadModelFile(context, "model.tflite")
        val options = Interpreter.Options().apply {
            addDelegate(FlexDelegate())  // TF ops 지원
        }
        interpreter = Interpreter(model, options)

        labels = loadLabelsFromAssets(context, "labels.json")
    }

    private fun loadModelFile(context: Context, filename: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun loadLabelsFromAssets(context: Context, filename: String): List<String> {
        val assetManager = context.assets
        val inputStream = assetManager.open(filename)
        val bufferedReader = BufferedReader(InputStreamReader(inputStream))
        val jsonString = bufferedReader.use { it.readText() }
        bufferedReader.close()

        val jsonObject = JSONObject(jsonString)
        val jsonArray = jsonObject.getJSONArray("labels")
        val labels = mutableListOf<String>()
        for (i in 0 until jsonArray.length()) {
            labels.add(jsonArray.getString(i))
        }
        return labels
    }

    fun predict(input: Array<Array<FloatArray>>): Int {
        val output = Array(1) { FloatArray(31) }
        interpreter.run(input, output)
        return output[0].indices.maxByOrNull { output[0][it] } ?: -1
    }
}
