# Hackathon Context — QBI Hackathon 2026

## Logistics
- **Dates:** June 27–28, 2026 (48 hours, Sat–Sun).
- **Venue:** UCSF Mission Hall, Room 1400, Mission Bay, San Francisco. 9:00 AM – 9:00 PM.
- **Hosts:** UCSF / QBI (director: Nevan Krogan) with UC Berkeley & UC Santa Cruz; supported by QB3.
- **Register:** Eventbrite + join the QBI Hackathon Slack workspace.
- **Mixers (team formation / data hunting):** Apr 27 (UCB) · May 14 (UCSC) · Jun 11 (UCSF).

## Agenda (both days)
- Sat: 9:00 check-in · 9:30 opening · 10:30 hack · 12:00 lunch · 1:00 hack · 6:00 dinner · 9:00 end.
- Sun: 9:00 check-in · 9:30 hack · 12:00 lunch · **4:30 submit projects** · 5:00 presentations · 6:00 dinner · 7:00 end.

## Scope / theme (verbatim emphasis from QBI)
> "Are you a scientist in life sciences with GBs of exciting imaging (microscopy, **cryoEM**, MRI, X-Ray, etc.) or mass spec data that you need help processing... Maybe you want to automatically **segment cells/organs/particles** or denoise and cluster your data?... generate a **user-friendly interface for some command line tools** you have been using?"

Core data types: **light microscopy, electron microscopy (cryo-EM), proteomics/mass-spec.** Methods of interest: computer vision, ML/DL, transformers, FFTs, topological data processing, GUIs over CLI tools.

## Rules that shape our build
- **Intellectual property:** all submissions are **open source under the MIT license** (mandatory). → repo is MIT.
- **Infrastructure provided:** workstations with **4× RTX 2080 Ti** + cloud compute time. → design for a single 11 GB GPU; offload heavy runs to cloud or pre-compute.
- **Team shape:** ~3 people — one scientist with data, one developer, one generalist. (We are Tony/Eva/Bindu.)
- **Prizes:** monetary + **long-term QBI support toward a publication** → pick something publishable and reusable.

## Judges & what they reward
Panel = technically-minded scientists + science-minded developers + VCs. → a **working, demo-able product** with **real numbers** beats a clever notebook.

## Why our project (A2 / CryoClear) fits the category
- "cryoEM" is named explicitly in the call.
- "segment ... particles" is exactly particle picking.
- "user-friendly interface for command line tools" = wrapping Topaz/CryoSegNet in an interactive GUI.
- cryo-EM is a UCSF crown jewel → also scores on "closer to UCSF research."
Keep the framing **scientist-and-data-first** ("auto-pickers drown you in junk") so the fit is unmistakable. The real-time/streaming angle is an extension, not a drift — it's still EM-image processing.
