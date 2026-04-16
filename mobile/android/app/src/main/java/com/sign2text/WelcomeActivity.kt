package com.sign2text

import android.content.Intent
import android.os.Bundle
import android.view.View
import androidx.activity.ComponentActivity
import com.google.android.material.button.MaterialButton
import com.google.android.material.dialog.MaterialAlertDialogBuilder

class WelcomeActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_welcome)

        findViewById<MaterialButton>(R.id.btnContinue).setOnClickListener {
            startActivity(Intent(this, MainActivity::class.java))
        }

        findViewById<MaterialButton>(R.id.btnCredits).setOnClickListener { button ->
            animateCreditsButton(button) { showCreditsDialog() }
        }
    }

    private fun animateCreditsButton(button: View, onAnimationEnd: () -> Unit) {
        button.animate()
            .scaleX(0.9f)
            .scaleY(0.9f)
            .alpha(0.85f)
            .setDuration(120)
            .withEndAction {
                button.animate()
                    .scaleX(1f)
                    .scaleY(1f)
                    .alpha(1f)
                    .setDuration(140)
                    .withEndAction(onAnimationEnd)
                    .start()
            }
            .start()
    }

    private fun showCreditsDialog() {
        val dialogView = layoutInflater.inflate(R.layout.dialog_credits, null)

        val dialog = MaterialAlertDialogBuilder(this)
            .setView(dialogView)
            .setBackground(android.graphics.drawable.ColorDrawable(android.graphics.Color.TRANSPARENT))
            .create()

        dialog.setOnShowListener {
            dialog.window?.apply {
                setDimAmount(0.6f)
                decorView.apply {
                    alpha = 0f
                    scaleX = 0.88f
                    scaleY = 0.88f
                    animate()
                        .alpha(1f)
                        .scaleX(1f)
                        .scaleY(1f)
                        .setDuration(280)
                        .setInterpolator(android.view.animation.DecelerateInterpolator())
                        .start()
                }
            }
        }

        dialog.show()
    }
}
