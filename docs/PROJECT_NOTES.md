# Results and next experiments

Status recorded on 2026-09-20. This page distinguishes operator observations, file checks and hypotheses so the project can grow without overstating its results.

## What has worked

The operator reports successful real-arm grasping and red-cube task completion after fine-tuning SmolVLA. Collection, dataset conversion, remote training, checkpoint transfer and local inference have all been exercised. A previous isolated test also loaded the real checkpoint and ran its preprocessing, inference and postprocessing on a recorded wrist frame, producing a finite seven-dimensional action.

The completed training run used 20,000 steps and batch size 64 on a rented RTX 4090. The reported final loss was about 0.013. There is no systematic trial count or independent success-rate estimate yet.

## White instruction, wrong object

Observed failure: when asked to put the white cube in the black box, the robot sometimes picks the red cube instead. Motion jitter has also been observed.

Read-only checks on the local files found:

- Excluding practice episode `000`, metadata contains 28 red and 27 white demonstrations, all marked `kept`. That flag records operator retention, not independently verified success.
- Raw sample counts are also similar: 26,879 red and 27,337 white, before conversion/trimming.
- Both saved white-trial traces contain the exact intended white instruction.
- Current inference code passes the configured task to the checkpoint preprocessor. Dataset conversion reads labels from `meta.json`.
- Layout prompts vary, but coverage per colour/layout combination is small and uneven. One recorded layout has five white episodes and no red episodes; another has six red and two white. Metadata describes the requested layout, not visual proof that the scene matched it.

These checks make a simple global colour-count imbalance or an accidentally red command in those two logs less likely. They do not establish that every demonstration was labelled correctly, or that the remote exported dataset preserved every label. That export is not available in this repository.

**Working hypothesis:** the policy has learned useful reaching/grasping behavior without reliably grounding the colour instruction in the current image. It may rely on spatial or scene correlations. Camera/viewpoint or lighting changes and training-label errors remain alternative explanations. This is not yet a confirmed root cause or evidence that SmolVLA cannot distinguish the colours. The [official SmolVLA guide](https://huggingface.co/docs/lerobot/v0.4.3/en/smolvla) describes language-conditioned actions and recommends sufficient demonstrations for each introduced scene variation.

A focused follow-up experiment:

1. Inspect exported task indices and representative videos against their labels.
2. Offline, hold image/state and sampling noise fixed; compare predicted action chunks for red versus white prompts. Repeat across several scenes. An action difference alone does not prove correct target selection.
3. After validating the controller stop paths, run matched trials from the same initial layout with each prompt, then swap the cubes' positions and repeat.
4. Record requested colour, first selected object, grasp outcome and placement outcome separately. Do not count “gripper holding something” as correct task completion.
5. Use the results to decide whether to add paired demonstrations, improve scene coverage or change training settings.

## Motion jitter

The current script includes target smoothing, optional temporal ensembling and gripper command handling. Their presence is not evidence that jitter is resolved. The two saved white-trial traces used smoothing 0.25 and had ensembling disabled.

Use recorded raw actions, smoothed targets, sent joint commands and measured state/torque to identify where oscillations first appear. Compare one change at a time and retain the working checkpoint and launch parameters as a baseline. Do not assume that more training alone will fix a controller or mechanical issue.

## Control code status

The source is preserved as an experimental working snapshot, not a validated safety controller. Static inspection still identifies issues requiring follow-up:

- Actions receive a new timestamp when published, including actions taken from an older chunk. Logged action age is not necessarily source-observation age.
- Reset/pause can overlap an in-flight prediction; episode generations are not used to reject older results.
- Collision checks occur after the current cycle's command sends.
- Ordinary exceptions do not all issue a stop/hold before cleanup.
- A stale-feedback hold uses cached joint angles as its target, which cannot establish the current physical pose.
- Automatic return-home is processed independently of `running`; `hold()` does not clear `going_home`. A pause/fault can therefore be followed by another return-home command. Its path does not pass through the complete policy-action gate sequence.

The earlier `Thread._stop` naming collision has been corrected in the current file. The other items above have not been certified on hardware. Keep the hardware stop accessible and validate these paths before relying on software pause or unattended operation.

## Portfolio description

> Built an end-to-end robot learning workflow on an AgileX Piper arm: SpaceMouse demonstration collection, wrist-camera/state synchronization, LeRobot dataset conversion, remote SmolVLA fine-tuning and local policy execution. Initial real-world trials achieved grasping and red-cube task completion. Ongoing work targets reliable language-conditioned colour selection, smoother motion and repeatable evaluation.

A selected [48-second real-arm demonstration](https://zhaoyufei.cn/projects/piper-smolvla-demo.mp4) and an [additional trial](https://zhaoyufei.cn/projects/piper-smolvla-additional-trial.mp4) are now linked from the README and personal project page. Copies are stored in `media/` in this repository. They are selected presentation recordings, not a formal evaluation set. Future updates can add measured trial outcomes and a setup diagram. The wording above intentionally avoids a success percentage or claims of general-purpose instruction following.
