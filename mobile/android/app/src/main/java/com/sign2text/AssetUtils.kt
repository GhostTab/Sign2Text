package com.sign2text

import android.content.Context
import java.io.BufferedReader
import java.io.InputStreamReader

object AssetUtils {
    fun readAssetText(context: Context, assetName: String): String {
        context.assets.open(assetName).use { input ->
            BufferedReader(InputStreamReader(input)).use { br ->
                return br.readText()
            }
        }
    }
}

