package com.sign2text

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class LetterAcceptanceTest {

    @Test
    fun weakCStaysBlank() {
        assertNull(LetterAcceptance.choose("C", 0.24f, "O", 0.20f, 2.8f, 2.4f, false, thumbIndexGap = 0.67f))
    }

    @Test
    fun closeSecondLetterDoesNotReplaceTheLeader() {
        assertNull(LetterAcceptance.choose("S", 0.42f, "T", 0.28f, 2.0f, 2.4f, false))
    }

    @Test
    fun clearBIsNotStolenByC() {
        assertEquals("B", LetterAcceptance.choose("B", 0.80f, "C", 0.10f, 0.8f, 2.4f, false))
    }

    @Test
    fun unsureOtherLetterStaysBlank() {
        assertNull(LetterAcceptance.choose("A", 0.40f, "S", 0.30f, 2.2f, 2.4f, false))
    }

    @Test
    fun clearEnoughUIsShown() {
        assertEquals("U", LetterAcceptance.choose("U", 0.56f, "R", 0.40f, 1.2f, 2.4f, false))
    }

    @Test
    fun weakUStaysBlank() {
        assertNull(LetterAcceptance.choose("U", 0.50f, "R", 0.40f, 1.4f, 2.4f, false))
    }

    @Test
    fun secondPlaceUDoesNotReplaceR() {
        assertEquals("R", LetterAcceptance.choose("R", 0.80f, "U", 0.10f, 0.7f, 2.4f, false))
    }

    @Test
    fun clearEnoughPIsShown() {
        assertEquals("P", LetterAcceptance.choose("P", 0.56f, "K", 0.40f, 1.2f, 2.4f, false))
    }

    @Test
    fun weakPStaysBlank() {
        assertNull(LetterAcceptance.choose("P", 0.50f, "K", 0.40f, 1.4f, 2.4f, false))
    }

    @Test
    fun secondPlacePDoesNotReplaceK() {
        assertEquals("K", LetterAcceptance.choose("K", 0.80f, "P", 0.10f, 0.7f, 2.4f, false))
    }

    @Test
    fun drawnZWins() {
        assertEquals("Z", LetterAcceptance.choose("D", 0.90f, "G", 0.05f, 0.4f, 2.4f, true))
    }

    @Test
    fun closedHandCalledCIsShownAsO() {
        assertEquals(
            "O",
            LetterAcceptance.choose("C", 0.80f, "O", 0.15f, 0.6f, 2.4f, false, thumbIndexGap = 0.18f),
        )
    }

    @Test
    fun openHandCalledOIsShownAsC() {
        assertEquals(
            "C",
            LetterAcceptance.choose("O", 0.70f, "C", 0.20f, 0.7f, 2.4f, false, thumbIndexGap = 0.67f),
        )
    }
}
