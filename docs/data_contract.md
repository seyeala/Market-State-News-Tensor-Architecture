# Data Contract

Every feature-bearing object must carry an `available_at` timestamp. A feature is valid for a decision only when `available_at <= decision_time`.
