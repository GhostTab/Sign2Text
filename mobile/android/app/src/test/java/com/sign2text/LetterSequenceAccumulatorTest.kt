package com.sign2text

import org.junit.Assert.assertEquals
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
}
