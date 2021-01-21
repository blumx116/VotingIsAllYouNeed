# Voting Is All You Need

`Voting Is All You Need` is a package for fast, seamless and quality group decision-making.
It balances the often-competing goals of giving everyone a voice while identifying strong decision-makers, resulting in an outcome that works well for everyone.
Additionally, `Voting Is All You Need` implements a clean and transparent process for understanding what decision was recommended.

# Functionality

`Voting Is All You Need` (`VIAYN`) works with groups - after adding people to the group, you can get to work.
To use `VIAYN` for a decision, simply create a question and a set of possible answers.
For each answer, people can predict how happy the group will be if that answer is chosen.

## Example
> Your organization wants to decide where to hold an event (we'll be fancy - let's say it's a gala). 
> You can create a question *"Where should we hold the gala?"*.
> Fill in some potential answers :
>  - *The Ritz*
>  - *Gotham Hall*
>  - *Central Park*
> You can also leave the answers open-ended, so people can submit more.
>    
> Members of the organization who want to help with planning can then predict how happy the attendees (or organization members) will be with the event if it's held at the Ritz versus Central Park.
> Based on these predictions, the app recommends a location to hold the event - let's say Central Park.
> After the event, people can vote about how happy they are with the event.
> If Central Park was a huge success, then we'll make note of the people who anticipated that.

The key is that when each user signs on, they're given a number of tokens.
Whenever that user signs on, they spend part of their balance - they can spend more if they're really confident about the quality of an answer.
`VIAYN` chooses which action to recommend based on the predicts made and how many tokens were submitted with each prediction.

After voting, everyone who submitted a prediction receives tokens proportional to the quality of their prediction and how many tokens they submitted.
Because of this, users who are better at predicting will accrue more tokens with which to influence decisions.
However, everyone's vote is always worth an equal amount, so everyone always has an equal voice about how much they like the outcomes.
Everyone can see what predictions were made and what votes were cast, but everything is completely anonymous.

Because of this duality, `VIAYN` addresses the following common problems:
 - **Groupthink** : members who can avoid falling prey to groupthink will accrue more tokens to help make better decisions.
 - **Collusion** : anonymity makes it difficult to collude - you never know if your co-conspirator backstabbed you.
 - **Inequality** : the solution with the best predicted outcome for everyone is chosen, regardless of how many tokens they have.
 - **Discrimination** : the algorithm knows 0 information about the personal data of each person and all decisions are anonymous, ensuring equitable treatment.
 - **Biases** : members who can avoid biases will get more tokens, to help make better decisions

# Code Organization

This code uses python with type hints.
To view the definition for most types used in the project, review `project_types.py`.
There are many different design decisions that can be made in the algorithm, so it is designed in a modular fashion, inheriting from the types in `project_types.py`.

For possible implementations of these classes, `samples/` has a folder of implementations.
>  - `samples/policy.py` contains policies for how a decision can be recommended from predictions
>  - `samples/payout.py` contains payout schemes for awarding tokens based on predictions and aggregate votes
>  - `samples/voting.py` and `samples/vote_ranges` contain different strategies for aggregating user votes
>  - `samples/agents.py` and `samples/env.py` are mostly useful for tests where AI agents are inserted in to the system

The preferred method of construction is using the factories in `samples/factory`.
A factory class corresponding to each of the above implementation parts is provided, with a `Config` type that can be passed in to it.

`experiments/` and `logging/` are for experiments with AI agents.
Both are undergoing a refactoring to make them more cleanly usable.

There are also 200+ tests that should always be passing on `main`.

Further documentation found at : https://blumx116.github.io/VotingIsAllYouNeed/build/html/index.html
