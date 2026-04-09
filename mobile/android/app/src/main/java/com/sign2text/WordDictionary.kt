package com.sign2text

/**
 * English word lookup from a line-based asset ([fromLines]). Supports exact match and prefix checks
 * for UI hints (e.g. "could still become a word").
 */
class WordDictionary private constructor(
    private val root: TrieNode
) {
    fun containsWord(word: String): Boolean {
        val w = word.trim().lowercase()
        if (w.isEmpty()) return false
        var n = root
        for (ch in w) {
            n = n.children[ch] ?: return false
        }
        return n.wordEnd
    }

    fun hasPrefix(prefix: String): Boolean {
        val p = prefix.trim().lowercase()
        if (p.isEmpty()) return true
        var n = root
        for (ch in p) {
            n = n.children[ch] ?: return false
        }
        return true
    }

    private class TrieNode(
        val children: MutableMap<Char, TrieNode> = mutableMapOf(),
        var wordEnd: Boolean = false
    )

    companion object {
        fun fromLines(text: String): WordDictionary {
            val root = TrieNode()
            for (line in text.lineSequence()) {
                val w = line.trim().lowercase()
                if (w.isEmpty()) continue
                if (!w.all { it in 'a'..'z' }) continue
                var n = root
                for (ch in w) {
                    n = n.children.getOrPut(ch) { TrieNode() }
                }
                n.wordEnd = true
            }
            return WordDictionary(root)
        }
    }
}
