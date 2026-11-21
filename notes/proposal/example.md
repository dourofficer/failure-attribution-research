# Future in the Past: Learning to Forecast from Evolving Evidence

## Abstract
Future prediction requires gathering relevant evidence and assessing its implications for an event’s outcome at a future time. However, although current agents can effectively gather evidence, they struggle to anticipate how the conditions it describes will evolve and how those changes will affect the event at the prediction horizon. Even human experts find this difficult, because evidence is interdependent and the future impact of an individual observation is hard to assess in isolation. To model these evolving dependencies, we propose a memory-based meta-learning framework that uses resolved outcomes to supervise forecasts across intermediate dates. Through this supervision, LLM agents learn how evidence evolves jointly and how its implications for future outcomes change over time, developing transferable strategies for updating event-specific memory and revising predictions. We investigate whether accumulated past information improves forecasting through event-specific memory, and whether the learned forecasting ability transfers to new events without prior event-specific memory.

## 1 Motivation and Research Questions
Current future prediction agents can gather relevant evidence, but struggle to anticipate how it will evolve and how those changes will affect an event at the prediction horizon. An observation’s predictive significance depends on subsequent developments and on its interactions with other evidence. An injury report, for example, has different implications depending on whether recovery occurs before a match or symptoms persist near kickoff.
Learning this ability requires data linking earlier evidence, its subsequent evolution, and the eventual outcome. Such data are difficult to obtain because relevant developments must be followed over time while preserving what was known at each date. Even human experts cannot readily annotate how interacting evidence will develop or what its future predictive significance will be. Outcome labels alone leave the intervening evolution of evidence unrecorded.
To address this limitation, we propose a memory-based meta-learning framework that uses resolved outcomes to supervise forecasts across intermediate dates. Through this supervision, LLM agents learn how evidence evolves jointly and how its implications for future outcomes change over time, developing transferable strategies for updating event-specific memory and revising predictions.
RQ1: Learning to use past information. How can learned memory updates integrate past and current evidence into a context that improves forecasting?
RQ2: Learning transferable forecasting ability. Can the framework learn forecasting skills that transfer to new events without prior event-specific memory?

## 2 Problem Setup
Forecasting task and information
An event q specifies K mutually exclusive and exhaustive outcomes, a target time Tq, and a resolution rule. The agent issues forecasts at ordered cutoffs τ₁, …, τD before the target and public resolution. At each τt, it produces a probability vector pt for the eventual outcome y using only sources and document versions available by that cutoff. It denotes the evidence retrieved at τt. The number of cutoffs D and their spacing are experimental variables.
Two research settings
The two settings share the target evaluation cutoff τD and the same event outcome. They distinguish the predictive value of accumulated event history from forecasting ability learned across events.
Research setting
Available information
Scientific question
Forecasting with history
Current evidence and accumulated event-specific memory
Does accumulated evidence history improve context quality and prediction?
Transfer without prior memory
Evidence available at the target cutoff; no carried event-specific memory
Does learned forecasting ability transfer to unseen events?


In the history-conditioned setting, past observations are retained as event-specific context. In the transfer setting, no memory or predictions are carried from earlier runs of that event. The agent may still retrieve older sources available by the cutoff and take notes during its current run. Parameters remain fixed during evaluation.


## Outcome and evaluation
B(pt, y) = k = 1K(pt,k − yk)2
The resolved label y is one-hot; pt is nonnegative and sums to one. We use the full-vector Brier loss for binary and multiclass events, with range 0–2. The same outcome scores every earlier forecast offline and never appears in historical inputs or memory. Proper scoring rewards calibrated uncertainty in expectation [1]. It does not imply that confidence should increase with time or that every realized forecast must improve. Both settings share the target cutoff; their different inference costs are reported.

## References
[1] Gneiting, T., and Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. JASA 102(477), 359–378.
