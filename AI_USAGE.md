# AI Usage

## Tools Used

- **ChatGPT:** Used to understand the task requirements, discuss implementation approaches, review code, and debug errors.
- **VS Code:** Used to edit and organize the project files.
- **uv:** Used to install dependencies and run the project.
- **pytest:** Used to run the provided public tests and additional validation tests.

## AI Suggestions Reviewed and Corrected

An initial cumulative-history approach could depend on the input row order. I reviewed and corrected the history calculation to order customer and timestamp groups before computing cumulative counts and failure rates.

I then tested this behavior by calculating history features on the original dataset and on a shuffled version of the same dataset. The results matched for every parcel.

Test result: `PASS: History features are independent of input row order`.

## Leakage Prevention

Only features available when a parcel is created are used for prediction. The target (`failed`), `delivered_at`, `courier_notes`, and `parcel_id` are excluded from model features. Customer history uses only strictly earlier parcels, and the model is evaluated using a chronological train-test split. The test set is not used to fit the model.
