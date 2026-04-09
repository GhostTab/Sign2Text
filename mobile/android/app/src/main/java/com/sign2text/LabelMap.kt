package com.sign2text

import org.json.JSONObject

data class LabelMap(val idToLabel: Map<Int, String>) {
    /** Must match TFLite output size and training `num_classes`. */
    val numClasses: Int get() = idToLabel.size

    fun labelFor(id: Int): String = idToLabel[id] ?: "?"

    companion object {
        fun fromJson(json: String): LabelMap {
            val obj = JSONObject(json)
            val map = mutableMapOf<Int, String>()
            val keys = obj.keys()
            while (keys.hasNext()) {
                val k = keys.next()
                map[k.toInt()] = obj.getString(k)
            }
            return LabelMap(map.toSortedMap())
        }
    }
}

