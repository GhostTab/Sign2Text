package com.sign2text

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class WordDictionaryTest {

    @Test
    fun containsWordAndPrefix() {
        val d = WordDictionary.fromLines(
            """
            hello
            help
            """.trimIndent()
        )
        assertTrue(d.containsWord("hello"))
        assertTrue(d.hasPrefix("hel"))
        assertTrue(d.hasPrefix("hello"))
        assertFalse(d.containsWord("hel"))
        assertFalse(d.hasPrefix("hex"))
    }

    @Test
    fun ignoresInvalidLines() {
        val d = WordDictionary.fromLines(
            """
            ok
            123bad
            """.trimIndent()
        )
        assertTrue(d.containsWord("ok"))
        assertFalse(d.containsWord("123bad"))
    }
}
