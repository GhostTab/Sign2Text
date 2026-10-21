package com.sign2text

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class LetterSequenceAccumulatorTest {

    @Test
    fun appendsOnTransitionToNewLetter() {
        val acc = LetterSequenceAccumulator()
        acc.onDebouncedLetter("H")
        acc.onDebouncedLetter("H")
        acc.onDebouncedLetter("H")
        assertEquals("H", acc.spelling)
        acc.onDebouncedLetter("?")
        acc.onDebouncedLetter("E")
        assertEquals("HE", acc.spelling)
    }

    @Test
    fun repeatsLetterAfterUncertainty() {
        val acc = LetterSequenceAccumulator()
        acc.onDebouncedLetter("L")
        acc.onDebouncedLetter("?")
        acc.onDebouncedLetter("L")
        assertEquals("LL", acc.spelling)
    }

    @Test
    fun ignoresQuestionMarkAndClearsNotApplied() {
        val acc = LetterSequenceAccumulator()
        acc.onDebouncedLetter("?")
        acc.onDebouncedLetter("?")
        assertEquals("", acc.spelling)
    }

    @Test
    fun clearResetsState() {
        val acc = LetterSequenceAccumulator()
        acc.onDebouncedLetter("A")
        acc.clear()
        acc.onDebouncedLetter("A")
        assertEquals("A", acc.spelling)
    }

    @Test
    fun takeSpellingAndClearIsAtomic() {
        val acc = LetterSequenceAccumulator()
        acc.onDebouncedLetter("H")
        assertEquals("H", acc.takeSpellingAndClear())
        assertEquals("", acc.spelling)
        acc.onDebouncedLetter("H")
        assertEquals("", acc.spelling)
        acc.onDebouncedLetter("?")
        acc.onDebouncedLetter("I")
        assertEquals("I", acc.spelling)
    }

    @Test
    fun saveThenSpellNextWordThenClear() {
        val acc = LetterSequenceAccumulator()
        acc.onDebouncedLetter("H")
        acc.onDebouncedLetter("I")
        val saved = acc.takeSpellingAndClear()
        assertEquals("HI", saved)
        assertEquals("", acc.spelling)

        acc.onDebouncedLetter("B")
        acc.onDebouncedLetter("Y")
        acc.onDebouncedLetter("E")
        assertEquals("BYE", acc.spelling)

        acc.clear()
        assertEquals("", acc.spelling)
    }

    @Test
    fun backspaceRemovesLastLetterAndAllowsRetype() {
        val acc = LetterSequenceAccumulator()
        acc.onDebouncedLetter("H")
        acc.onDebouncedLetter("E")
        acc.onDebouncedLetter("L")
        assertEquals("HEL", acc.spelling)

        assertTrue(acc.deleteLastLetter())
        assertEquals("HE", acc.spelling)

        // Can re-add the same letter after backspace (transition state reset)
        acc.onDebouncedLetter("L")
        assertEquals("HEL", acc.spelling)
        acc.onDebouncedLetter("L")
        assertEquals("HEL", acc.spelling)
        acc.onDebouncedLetter("O")
        assertEquals("HELO", acc.spelling)
    }

    @Test
    fun backspaceOnEmptyReturnsFalse() {
        val acc = LetterSequenceAccumulator()
        assertFalse(acc.deleteLastLetter())
        assertEquals("", acc.spelling)
    }
}
