# Books of Gold

A learning project with an Express/MongoDB API (`Server/`), a React storefront (`front/`), and a React dashboard scaffold (`dashboard/`).

1. Create a local MongoDB database for the demo. In `Server/`, copy `.env.example` to `.env`, set a random `JWT_SECRET`, then run `npm ci`, `npm run seed`, and `npm start`.
2. In `front/`, copy `.env.example` to `.env.local`, then run `npm ci` and `npm start`.
3. For the dashboard scaffold, copy its `.env.example` to `.env.local`, then run `npm ci` and `npm start` from `dashboard/`.

The API listens on loopback. The seed command adds demo products without resetting the database or creating default accounts. The checkout UI uses Google Pay test mode. Use a disposable database and local development settings.

The original React tooling and lockfiles are preserved. Use an isolated Node 16 environment for the legacy frontend dependencies.
