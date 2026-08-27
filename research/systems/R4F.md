# R4F / Flash-Next

**Status:** `EXPERIMENTAL`

R4F is the architecture-aware Flash-Next track: choose tensor precision,
packing, and execution strategy together with the target kernel and memory
hierarchy. The research asks whether exposed bytes, not nominal bit width,
are the limiting resource at the relevant tile and sequence shapes.

The public bring-up boundary is [`research/r4f`](../r4f/) and
[`research/flash-next`](../flash-next/). The source register marks the
full-model and historical-receipt material as incomplete or omitted. No
R4F model payload is bundled, and the track is not an alternate foreign
execution backend for HAR.

Open work includes complete tensor coverage, representative quality tests,
kernel correctness, and end-to-end measurements with caller-supplied models.
