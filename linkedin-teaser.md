# LinkedIn teaser (draft — edit voice, then post once the blog is live)

A small side study, mostly for fun: every RLHF post mentions that DPO's
closed-form solution is a Gibbs distribution — but almost none write down the
actual term-by-term dictionary between the two fields.

So I did: energy ↔ negative reward, entropy ↔ (relative!) KL to the reference
policy, temperature ↔ DPO's β — with the convention trap that β acts like k_BT,
not inverse temperature. Then I tried to verify the free-energy limit
empirically on GSM8K and learned exactly why that's harder than the clean math
suggests.

Full post (with the parts that *didn't* work): [LINK]
