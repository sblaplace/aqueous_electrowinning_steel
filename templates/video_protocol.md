# Video Recording Protocol

Continuous video recording is mandatory for every plating experiment.
Video is the fastest way to answer questions that arise weeks later:
"was there a bubble on the cathode?", "did the stirrer stop?", "what
color was the deposit at t=1200s?".

## Camera positions

| Position     | Field of view                          | Mount                        |
|-------------|----------------------------------------|------------------------------|
| `overhead`  | Cell interior + bath surface           | Clamped above the cell       |
| `panel`     | Cathode face (or Hull panel)           | Angle bracket on cell rim    |
| `instruments` | Power supply display, multimeter,    | Tripod behind the bench      |
|             | coulomb counter readout                |                              |
| `wide`      | Full bench: cell, instruments, operator | Fixed corner of the bench   |

### Camera requirements

- **Resolution**: ≥ 1080p (1920 × 1080).
- **Duration**: Must cover the full experiment with no gaps.
  Use AC power or a battery pack that lasts 2× the expected run.
- **Timestamp**: Enable on-screen clock if available.  Even without
  it, the `video_index.csv` links filenames to `timestamp_s`.
- **Lighting**: Avoid glare on the cathode face.  A polarizing filter
  on the `panel` camera helps if specular reflection is a problem.

## video_index.csv

Every experiment directory includes a `video_index.csv` that maps
camera filenames to experiment timestamps and events.

| Column       | Unit | Description                                            |
|-------------|------|--------------------------------------------------------|
| `timestamp_s`| s   | Experiment timestamp (matches timeseries CSV)          |
| `camera`     | —   | Camera position: `overhead`, `panel`, `instruments`, `wide` |
| `filename`   | —   | Video file name (relative to `video/` directory)       |
| `event`      | —   | Free-text: "start", "stop", "sample taken", etc.       |

### Example

```
timestamp_s,camera,filename,event
0,overhead,overhead_0001.mp4,start
0,panel,panel_0001.mp4,start
0,instruments,instruments_0001.mp4,start
0,wide,wide_0001.mp4,start
600,panel,panel_0001.mp4,sample extracted from cathode
1800,overhead,overhead_0001.mp4,stop
1800,panel,panel_0001.mp4,stop
1800,instruments,instruments_0001.mp4,stop
1800,wide,wide_0001.mp4,stop
```

## Storage layout

```
experiments/data/<run_id>/
├── manifest.json
├── timeseries.csv
├── mass_log.csv
├── video_index.csv
└── video/
    ├── overhead_0001.mp4
    ├── panel_0001.mp4
    ├── instruments_0001.mp4
    └── wide_0001.mp4
```

## Procedure

1. **Before power-on**: Start all cameras.  Verify recording is active
   by checking the red LED or on-screen indicator on each camera.
2. **Log video_index.csv**: Record the start event at `timestamp_s = 0`.
3. **During experiment**: If you extract a sample, adjust stirring, or
   change any physical condition, add a row to `video_index.csv` with
   the current `timestamp_s`.
4. **After power-off**: Stop all cameras.  Log the stop event.
5. **File transfer**: Copy video files to `video/` within one hour of
   experiment end.  Do not delete camera originals until the transfer
   is verified (file size > 0, plays back without error).
6. **Manifest**: Set `video.recording_status` in `manifest.json` to
   `"complete"`, `"partial"`, or `"none"` (with justification).

## Failure handling

- If a camera fails mid-run, log the failure in `video_index.csv` with
  an `event` like "camera_battery_died" and the last recorded
  `timestamp_s`.
- Set `recording_status = "partial"` in the manifest.
- A run with **no** video is still valid data, but `recording_status`
  must be `"none"` and a justification is mandatory in `video.notes`.
