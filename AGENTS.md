# Project instructions

## Style

Follow the applicable Google style guide for every authored file:

- Python: <https://google.github.io/styleguide/pyguide.html>.
- JavaScript: <https://google.github.io/styleguide/jsguide.html>.
- TypeScript: <https://google.github.io/styleguide/tsguide.html>.
- HTML and CSS: <https://google.github.io/styleguide/htmlcssguide.html>.
- Shell: <https://google.github.io/styleguide/shellguide.html>.
- Markdown: <https://google.github.io/styleguide/docguide/style.html>.
- Documentation: <https://developers.google.com/style>.

Use two-space indentation for JSON. Where Google has no format-specific guide,
follow the format's conventions and Google's naming and clarity principles.
Keep generated data reproducible and record the generator and configuration.

## Design

Read docs/design.md before implementation. Keep inference, attention selection,
sonification, and rendering separate. Use one trace as the source for audio and
visual events. Distinguish query time, attended source time, and playback time.

Do not label chunked attention as full-context attention. Record checkpoint
revisions, time mappings, thresholds, and selection rules in exported traces.
