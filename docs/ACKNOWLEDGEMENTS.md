# Acknowledgements

HAR and the surrounding research are independent local work. The project
does not represent or speak for the upstream model, systems, or research
projects listed here.

## Attribution / prior work

This project has been influenced by a lot of open-source work, papers,
discussions and experiments over time, and I may not remember every source
that shaped an idea or implementation.

I’ve made a good-faith effort to identify and credit the projects and people I
know influenced the work. If you recognize something that should be cited,
attributed, or linked and it’s missing, please open an issue. I’ll investigate
it and update the provenance record accordingly.

Inclusion here doesn’t imply that every idea is derived from the cited work,
and absence doesn’t imply a claim of originality.

- The Qwen team is credited for the Qwen model family and the architecture
  conventions used as a compatibility reference. HAR's Rust implementation is
  separate from the model authors' work.
- The authors and maintainers of FreeToken, colibri, dflash, Medusa, and
  LayerSkip are credited for ideas and public technical references around
  expert movement, residency, and speculative decoding.
- Khronos is credited for the Vulkan API and specification. The Vulkan driver
  remains a platform boundary; HAR does not redistribute driver code.
- The maintainers of ash, serde, serde_json, sha2, hex, thiserror, libc, and
  regex are credited through their Cargo packages and licenses.
- The maintainers of safetensors and Transformers are credited for public
  tensor/model-format and architecture references.
- AMD and Mesa maintain the public hardware and RADV documentation used to
  describe the target boundary.

See [PROVENANCE.md](../PROVENANCE.md) and
[docs/references.md](references.md) for source URLs, license decisions, and
the distinction between inspiration, reference, and copied code.

Development of this candidate used AI assistance alongside human review.
Generated suggestions are not treated as external authorship or as evidence
for a performance result.
