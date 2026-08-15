import jax
import jax.numpy as jnp


def make_model(m):
    """Return BF_ewa_model closure bound to BF instance m using Categorical likelihood."""

    def BF_ewa_model(K, J, tech, id, bout, y_prev, s, ps, ks, pr, co, yo, age):
        # --- Priors ---
        lam = m.dist.exponential(
            name="lambda", rate=1.0
        )  # Multinomial sensitivity (inverse temperature)
        mu = m.dist.normal(
            name="mu", loc=jnp.zeros(8), scale=1.0
        )  # Population-level means for the 8 learning effects
        sigma = m.dist.exponential(
            name="sigma", rate=jnp.full(8, 3.0)
        )  # SDs for the 8 learning effects
        L_Rho = m.dist.lkj_cholesky(
            name="L_Rho", dimension=8, concentration=3.0
        )  # Cholesky factor of correlation matrix
        z = m.dist.normal(
            name="z", loc=jnp.zeros((8, J)), scale=1.0
        )  # Individual z-scores (non-centered parameterization)
        b_age = m.dist.normal(
            name="b_age", loc=jnp.zeros(2), scale=1.0
        )  # Age slopes for phi and gamma

        # --- Individual-level varying effects (non-centered) ---
        L_scaled = L_Rho * sigma[:, None]  # Scale the Cholesky factor by the SDs
        a_id = (L_scaled @ z).T  # Individual deviations from population means (J × 8)

        # --- Per-observation parameter arrays (N-length vectors) ---
        phi_arr = jax.nn.sigmoid(
            mu[0] + a_id[id, 0] + b_age[0] * age
        )  # Attraction updating weight (0→strong memory, 1→recent payoff)
        gamma_arr = jax.nn.sigmoid(
            mu[1] + a_id[id, 1] + b_age[1] * age
        )  # Social learning weight (0→individual, 1→social)
        fconf_arr = jnp.exp(
            mu[2] + a_id[id, 2]
        )  # Conformity exponent (frequency dependence strength)
        beta_payoff = mu[3] + a_id[id, 3]  # Payoff-bias coefficient
        beta_kin = mu[4] + a_id[id, 4]  # Kin-bias coefficient
        beta_rank = mu[5] + a_id[id, 5]  # Rank-bias coefficient (alpha status)
        beta_cohort = mu[6] + a_id[id, 6]  # Cohort-bias coefficient (age similarity)
        beta_age_bias = mu[7] + a_id[id, 7]  # Age-bias (YOB) coefficient

        def step_fn(carry, x):
            """Process one observation: update attraction scores and compute choice probability vector."""
            attraction_scores = carry

            # --- Unpack per-observation inputs ---
            (
                phi_i,  # Attraction updating weight for this observation
                gamma_i,  # Social learning weight for this observation
                fconf_i,  # Conformity exponent for this observation
                beta_pay_i,  # Payoff-bias coefficient
                beta_kin_i,  # Kin-bias coefficient
                beta_rank_i,  # Rank-bias coefficient
                beta_cohort_i,  # Cohort-bias coefficient
                beta_age_bias_i,  # Age-bias (YOB) coefficient
                tech_chosen,  # Index of the technique chosen (0-indexed)
                individual_id,  # Index of the individual (0-indexed)
                bout_index,  # Foraging bout number (1 = first bout → reset attractions)
                prev_yields,  # Previous personal yields for each technique (K,)
                freq_cue,  # Frequency cue: social observation counts per technique (K,)
                payoff_cue,  # Payoff-bias cue per technique (K,)
                kin_cue,  # Kin-bias cue per technique (K,)
                rank_cue,  # Rank-bias cue per technique (K,)
                cohort_cue,  # Cohort-bias cue per technique (K,)
                age_bias_cue,  # Age-bias (YOB) cue per technique (K,)
            ) = x

            # --- Step 1: Update attraction scores (EWA rule) ---
            # First bout → reset to zero; otherwise → exponential moving average of past yields
            attraction_new = jnp.where(
                bout_index == 1,
                jnp.zeros(K),
                (1.0 - phi_i) * attraction_scores[individual_id] + phi_i * prev_yields,
            )

            # --- Step 2: Individual learning probability (softmax over attractions) ---
            prob_individual = jax.nn.softmax(lam * attraction_new)

            # --- Step 3: Social learning probability ---
            # 3a. Log-linear model for non-frequency social cues (techniques 2..K relative to technique 1)
            social_cue_weights = jnp.exp(
                beta_pay_i * payoff_cue[1:]
                + beta_kin_i * kin_cue[1:]
                + beta_rank_i * rank_cue[1:]
                + beta_cohort_i * cohort_cue[1:]
                + beta_age_bias_i * age_bias_cue[1:]
            )
            # 3b. Combine with frequency cue (raised to conformity exponent)
            social_weights = jnp.concatenate(
                [jnp.ones(1), social_cue_weights]
            ) * jnp.power(freq_cue, fconf_i)
            # 3c. Normalize to get social learning probability
            safe_denom = jnp.where(
                jnp.sum(social_weights) > 0, jnp.sum(social_weights), 1.0
            )
            prob_social = social_weights / safe_denom

            # --- Step 4: Mix individual and social learning ---
            prob_mixed = (1.0 - gamma_i) * prob_individual + gamma_i * prob_social

            # --- Step 5: Choice probability vector ---
            # Use mixed probability only when social information is available (bout > 1 AND observed demonstrations)
            has_social_info = jnp.logical_and(bout_index > 1, jnp.sum(freq_cue) > 0)
            prob_final = jnp.where(
                has_social_info, prob_mixed, prob_individual
            )

            updated_attractions = attraction_scores.at[individual_id].set(
                attraction_new
            )
            return updated_attractions, prob_final

        # --- Run the scan over all observations ---
        attraction_init = jnp.zeros((J, K))
        xs = (
            phi_arr,
            gamma_arr,
            fconf_arr,
            beta_payoff,
            beta_kin,
            beta_rank,
            beta_cohort,
            beta_age_bias,
            tech,
            id,
            bout,
            y_prev,
            s,
            ps,
            ks,
            pr,
            co,
            yo,
        )
        _, probs_matrix = jax.lax.scan(step_fn, attraction_init, xs)
        m.dist.categorical(name="obs", probs=probs_matrix, obs=tech)

    return BF_ewa_model
