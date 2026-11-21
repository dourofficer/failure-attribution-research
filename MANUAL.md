How this project is run
=======================

This manual is for you, the researcher. The agent has its own rulebook in CLAUDE.md, which says the same things as instructions. Read this once; it takes about ten minutes.


The idea
--------

GPU time is expensive and easy to spend on the wrong question. So the project works in campaigns. A campaign is one research question plus the experiments that answer it, and nothing runs until four things are written down:

    Question         the claim the work tests
    Decision         what you will do differently depending on the outcome
    Done condition   the number, table or artifact that ends the work
    Kill bar         the result that ends this line for good, with no rescue runs

For open-ended work you may set the done condition and the kill bar to "None". That is your call; the agent will not do it for you.

You make every decision that is hard to undo. The agent does the rest: it argues with you during the discussion, writes the code, tests it, runs it once you approve, and writes down what happened.


The workflow
------------

Lines marked [YOU] wait for you. Lines marked "agent" happen without asking.

    +----------------------------------------------------------------+
    | EVERY SESSION, the agent first reads                           |
    |          docs/STATE.md             what is active right now    |
    |          docs/decisions/index.md   what is already decided     |
    +----------------------------------------------------------------+
                                    |
                                    v
    +----------------------------------------------------------------+
    | 1. DISCUSS                                                     |
    +----------------------------------------------------------------+
    | [YOU]    bring a question or an idea                           |
    | agent    sharpens it, says where it could break,               |
    |          proposes experiments                                  |
    +----------------------------------------------------------------+
                                    |
                                    |   the four items are settled
                                    v
    +----------------------------------------------------------------+
    | 2. CAMPAIGN                                                    |
    +----------------------------------------------------------------+
    | [YOU]    type /campaign                                        |
    | agent    shows a short overview                                |
    | [YOU]    agree or correct it                                   |
    | agent    writes the campaign README and opens a node           |
    |          in the decision graph                                 |
    +----------------------------------------------------------------+
                                    |
                                    v
    +----------------------------------------------------------------+
    | 3. BUILD + SMOKE                                               |
    +----------------------------------------------------------------+
    | agent    writes each experiment's code, config and tests,      |
    |          then rehearses it on a handful of items               |
    +----------------------------------------------------------------+
                                    |
                                    v
    +----------------------------------------------------------------+
    | 4. RUN CARDS                                                   |
    +----------------------------------------------------------------+
    | agent    hands you one card per experiment: command,           |
    |          cards, hours, outputs, smoke result                   |
    | [YOU]    say "go", or change something                         |
    +----------------------------------------------------------------+
                                    |
                                    v
    +----------------------------------------------------------------+
    | 5. RUN                                                         |
    +----------------------------------------------------------------+
    | agent    launches the independent experiments in parallel      |
    |                                                                |
    |          +-----------+   +-----------+   +-----------+         |
    |          |   exp A   |   |   exp B   |   |   exp C   |         |
    |          | cards 0-3 |   | cards 4-7 |   |  after A  |         |
    |          +-----+-----+   +-----+-----+   +-----+-----+         |
    |                |               |               |               |
    |                v               v               v               |
    |             worklog         worklog         worklog            |
    |                                                                |
    | agent    brings C's run card to you once A has finished        |
    | [YOU]    say "go" again                                        |
    +----------------------------------------------------------------+
                                    |
                                    |   done condition met, or kill bar hit
                                    v
    +----------------------------------------------------------------+
    | 6. CLOSE                                                       |
    +----------------------------------------------------------------+
    | agent    reports the result and suggests /close                |
    | [YOU]    type /close and agree the ruling                      |
    | agent    sets the node to "worked" or "failed"                 |
    +----------------------------------------------------------------+
                                    |
                                    v
    +----------------------------------------------------------------+
    | 7. COMMIT                                                      |
    +----------------------------------------------------------------+
    | agent    proposes a commit message                             |
    | [YOU]    say "commit"                                          |
    +----------------------------------------------------------------+
                                    |
                                    +-----> back to 1, with the graph
                                            one node richer

If you trust a campaign to run on its own, tell the agent "run this campaign without asking". It then launches later batches itself, as long as they stay inside the campaign's cards and kill bar. It still reports at a kill bar and still waits for you at step 6.


What always waits for you
-------------------------

    opening a campaign, and closing one
    launching runs (unless you switched that off for the campaign)
    commits
    using more cards than the campaign has, or cards someone else is using
    changing a question, a bar, a metric, or a setup file that a result already cites
    deleting results or model weights
    stopping, pausing or editing anything a live run uses
    anything in notes/, papers/ or .claude/settings

At a kill bar the agent stops that line and tells you. It does not try a rescue.


Where things live
-----------------

    MANUAL.md        this file
    CLAUDE.md        the agent's rules
    notes/           your own notes. The agent reads or writes here only when you ask.
    papers/          papers you want to keep at hand. The agent reads one only when you ask.
    data/            data shared by all campaigns and experiments, stored once
    docs/
      STATE.md       what is active, what is running, what waits for you, what is next
      ideas/         free-form ideas, nothing agreed
      campaigns/     one dated folder per campaign: README.md and worklogs/
      decisions/     the decision graph: one node per campaign
    setup/           shared datasets, models and metrics, one yaml file each
    src/
      experiments/<name>/      the experiment's code, README and tests
      utils/                   code shared between experiments
    scripts/
      experiments/<name>/      script.sh and config.yaml
    assets/
      experiments/<name>/      outputs, logs, the resolved config

Every experiment has the same name under src/, scripts/ and assets/. To ship one, copy those three folders and the setup files its config names.

There are no throwaway experiments. A quick check is a row in an existing campaign, or a small campaign of its own.


The records
-----------

Three things are written down, each in one place.

A worklog, one per experiment, kept in its campaign folder. Each run adds an entry: the run card, what was run, the result and where it is stored, what it means, and what it rules out. Failed runs are written up the same way as successful ones. The "rules out" part is the most valuable line in the project: it is what keeps you from paying for the same dead end twice.

A campaign README, which holds the four items, the table of experiments with their status, and at the end a closing summary.

A decision node, one per campaign. It is created as "open" when the campaign starts and holds the question. When you close the campaign it becomes "worked" or "failed" and holds the ruling. The nodes link to the nodes they followed from, which makes a graph. docs/decisions/index.md lists them all. The agent reads it before proposing anything, and so should you.

There are no separate handoff documents. docs/STATE.md is rewritten as things change, and its "Next step" section is what the next session picks up.


Shared setups
-------------

Many experiments use the same dataset, model or metric. Each of these is described once, in a yaml file under setup/, and has an id, which is its file name:

    setup/dataset/example_dev_v1.yaml
    setup/model/example_lm_v1.yaml
    setup/metric/auroc_grouped5_v1.yaml

An experiment's config names the setups it uses and adds only its own parameters:

    setup:
      dataset: example_dev_v1
      model: example_lm_v1
      metric: auroc_grouped5_v1
    params:
      layers: all

Two rules make ids trustworthy:

    An experiment cannot override a setup field. A different setup is a different id.
    A setup file is frozen once a run has used it. To change it, copy it to a new id (_v2).

So two experiments with the same ids were run under the same conditions, and you can compare them without opening any files. Each run also saves the fully expanded config next to its outputs.

    uv run python -m src.utils.config uses model example_lm_v1    which experiments use this model
    uv run python -m src.utils.config check                       was a used setup file edited later

You can add more kinds of setup by adding a folder, for example setup/prompt/.


Running experiments in parallel
-------------------------------

The campaign's experiment table has an "After" column. Experiments with nothing pending there can start together. One agent session launches each in its own tmux session on its own cards, waits for all of them, and writes the worklogs. They cannot collide, because each writes only to its own folder under assets/.

A single experiment also spreads over several cards by itself: its script splits the data into shards, one process per shard.

Every script can be rehearsed with SMOKE=1, can be rerun after a crash without redoing finished work, and leaves a .done or .failed file when it ends. You can close your laptop while it runs.


A session, start to finish
--------------------------

    You:    I think the probe is reading the wrong layer.
    Agent:  We read the final layer by default and never tested it. I suggest a sweep over
            all 32 layers. Folds must be grouped by source document or every layer will
            look too good. Kill bar: if no layer beats the final one by more than 0.02
            AUROC, we keep the default and stop.
    You:    Agreed.
    Agent:  This is ready for /campaign.
    You:    /campaign
    Agent:  (overview: one experiment, node D001, the four items)
    You:    Fine.
    Agent:  (writes the campaign and the node; builds and smokes the experiment)
            Run card: all 32 layers, 8 cards, about 25 minutes. Smoke passed.
    You:    Go.
    Agent:  (launches, waits, writes the worklog)
            Layer 16 reaches 0.84 against 0.68 at the final layer. That is far past the
            kill bar. It rules out "later layers are better". This is ready for /close.
    You:    /close
    Agent:  (overview: worked; ruling: read the probe at layer 16)
    You:    Yes.
    Agent:  (writes the closing summary and the ruling)
            Suggested commit: "Probe layer sweep: layer 16 beats the final layer".
    You:    Commit.

The files this session leaves behind are in the repo as an example: the campaign under docs/campaigns/, the node under docs/decisions/nodes/, and the experiment probe_layer_sweep. Delete them once you have real ones.


Starting a new project from this template
-----------------------------------------

    1. Copy the folder and run "git init" in the copy.
    2. Replace the placeholders: [RESEARCHER]       your name, in CLAUDE.md, docs/STATE.md and the close skill [PROJECT TOPIC]    in CLAUDE.md and README.md [PROJECT NAME]     in README.md, and the name in pyproject.toml
    3. Ask the agent to follow SETUP_GUIDE.md. It installs the environment and checks it with a real forward pass on the GPU. Delete the guide afterwards.
    4. Put your first ideas in docs/ideas/ and start a discussion.

The project uses uv for everything: "uv sync" to install, "uv run pytest -q" to test. Never call python or pip directly.
